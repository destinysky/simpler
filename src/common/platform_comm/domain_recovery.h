/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

enum class DomainRecoveryImpact : uint32_t {
    EXECUTION_ONLY = 0,
    DOMAIN_BINDINGS = 1,
    DOMAIN_TRANSPORT = 2,
    SHARED_COMMUNICATION = 3,
    DEVICE = 4,
};

enum class DomainRecoveryEngine : uint32_t { NONE = 0, SDMA = 1, URMA = 2, CCU = 3 };

enum class DomainRecoveryStatus : uint32_t {
    PRESERVED = 0,
    REBUILT = 1,
    ALREADY_RECOVERED = 2,
    INVALID_STATE = 3,
    STALE_GENERATION = 4,
    BUSY = 5,
    QUIESCENCE_UNPROVEN = 6,
    UNSUPPORTED = 7,
    GIVE_UP = 8,
    RECOVERY_FAILED = 9,
};

// A nonzero collective_quiescence asserts that the owner has fenced every
// domain user on every member, including async communication outside PTO.
// A local FIFO drain or a diagnostic queue snapshot alone is insufficient.
struct DomainRecoveryRequest {
    uint32_t abi_version{1};
    uint32_t struct_bytes{sizeof(DomainRecoveryRequest)};
    uint64_t recovery_id{0};
    uint64_t expected_generation{0};
    uint64_t failed_run_epoch{0};
    DomainRecoveryImpact impact{DomainRecoveryImpact::EXECUTION_ONLY};
    uint32_t collective_quiescence{0};
    int32_t original_rc{0};
    uint32_t reserved{0};
};

struct DomainRecoveryResult {
    uint32_t abi_version{1};
    uint32_t struct_bytes{sizeof(DomainRecoveryResult)};
    DomainRecoveryStatus status{DomainRecoveryStatus::INVALID_STATE};
    DomainRecoveryEngine engine{DomainRecoveryEngine::NONE};
    int32_t original_rc{0};
    int32_t recovery_rc{0};
    uint64_t recovery_id{0};
    uint64_t allocation_id{0};
    uint64_t previous_generation{0};
    uint64_t generation{0};
    uint64_t communication_context{0};
    uint64_t hccl_comm{0};
    uint64_t comm_stream{0};
    uint64_t device_ctx{0};
    uint64_t local_window_base{0};
    uint64_t window_size{0};
    uint64_t workspace{0};
    uint32_t domain_preserved{0};
    uint32_t rebind_required{0};
};

static_assert(std::is_standard_layout_v<DomainRecoveryRequest> && std::is_trivially_copyable_v<DomainRecoveryRequest>);
static_assert(std::is_standard_layout_v<DomainRecoveryResult> && std::is_trivially_copyable_v<DomainRecoveryResult>);
static_assert(sizeof(DomainRecoveryRequest) == 48 && offsetof(DomainRecoveryRequest, impact) == 32);
static_assert(sizeof(DomainRecoveryResult) == 120 && offsetof(DomainRecoveryResult, device_ctx) == 80);

// Optional host-runtime symbol. Backends without this operation retain their
// existing comm ABI and return UNSUPPORTED through ChipWorker.
extern "C" int comm_recover_domain(
    void *handle, uint64_t allocation_id, const DomainRecoveryRequest *request, DomainRecoveryResult *result
);

class DomainRecoveryManager {
public:
    enum class Phase : uint32_t { PREPARE = 1, BUILD = 2, RETIRE = 3 };
    struct Operations {
        bool preserve_supported{true};
        bool rebuild_supported{false};
        bool transport_rebuild_supported{false};
        // Agreement includes this rank's failure vote; no rank may publish a
        // replacement unless every member has completed every phase.
        std::function<int(Phase, int)> agree;
        std::function<int(DomainRecoveryResult &)> build;
        std::function<int()> retire;
        std::function<void()> discard;
    };

    uint64_t generation() const { return generation_; }
    void quarantine(int rc) {
        failed_ = true;
        last_result_.status = DomainRecoveryStatus::RECOVERY_FAILED;
        last_result_.recovery_rc = rc;
        last_result_.device_ctx = 0;
        last_result_.local_window_base = 0;
        last_result_.workspace = 0;
    }

    DomainRecoveryResult
    recover(const DomainRecoveryRequest &request, DomainRecoveryResult current, const Operations &operations) {
        current.recovery_id = request.recovery_id;
        current.original_rc = request.original_rc;
        current.previous_generation = generation_;
        current.generation = generation_;
        current.domain_preserved = 0;
        current.rebind_required = 0;
        auto reject = [&](DomainRecoveryStatus status, int rc = 0) {
            current.status = status;
            current.recovery_rc = rc;
            return current;
        };
        if (request.abi_version != 1 || request.struct_bytes != sizeof(DomainRecoveryRequest)) {
            return reject(DomainRecoveryStatus::UNSUPPORTED);
        }
        if (request.recovery_id == 0 || request.reserved != 0 || request.collective_quiescence > 1) {
            return reject(DomainRecoveryStatus::INVALID_STATE);
        }
        if (last_id_ == request.recovery_id && same_request(request, last_request_)) {
            auto result = last_result_;
            if (result.status == DomainRecoveryStatus::PRESERVED || result.status == DomainRecoveryStatus::REBUILT) {
                result.status = DomainRecoveryStatus::ALREADY_RECOVERED;
            }
            return result;
        }
        if (request.expected_generation != generation_ || request.recovery_id <= last_id_) {
            return reject(DomainRecoveryStatus::STALE_GENERATION);
        }
        if (failed_) return reject(DomainRecoveryStatus::GIVE_UP);
        if (request.impact == DomainRecoveryImpact::DEVICE ||
            request.impact == DomainRecoveryImpact::SHARED_COMMUNICATION ||
            (request.impact == DomainRecoveryImpact::DOMAIN_TRANSPORT && !operations.transport_rebuild_supported)) {
            failed_ = true;
            auto result = reject(DomainRecoveryStatus::GIVE_UP);
            remember(request, result);
            return result;
        }
        if (request.impact != DomainRecoveryImpact::EXECUTION_ONLY &&
            request.impact != DomainRecoveryImpact::DOMAIN_BINDINGS &&
            request.impact != DomainRecoveryImpact::DOMAIN_TRANSPORT) {
            return reject(DomainRecoveryStatus::INVALID_STATE);
        }
        if (request.impact == DomainRecoveryImpact::EXECUTION_ONLY) {
            if (!operations.preserve_supported) return reject(DomainRecoveryStatus::UNSUPPORTED);
            current.status = DomainRecoveryStatus::PRESERVED;
            current.domain_preserved = 1;
            remember(request, current);
            return current;
        }
        if (!operations.rebuild_supported) return reject(DomainRecoveryStatus::UNSUPPORTED);
        if (!request.collective_quiescence) return reject(DomainRecoveryStatus::QUIESCENCE_UNPROVEN);
        if (generation_ == std::numeric_limits<uint64_t>::max()) return reject(DomainRecoveryStatus::GIVE_UP);
        if (!operations.agree || !operations.build || !operations.retire || !operations.discard) {
            return reject(DomainRecoveryStatus::UNSUPPORTED);
        }
        auto checked = [](const auto &call) {
            try {
                return call();
            } catch (...) {
                return -1;
            }
        };
        DomainRecoveryResult replacement = current;
        int rc = checked([&] {
            return operations.agree(Phase::PREPARE, 0);
        });
        if (rc == 0) {
            int local_rc = checked([&] {
                return operations.build(replacement);
            });
            int agreement_rc = checked([&] {
                return operations.agree(Phase::BUILD, local_rc);
            });
            rc = local_rc != 0 ? local_rc : agreement_rc;
        }
        if (rc == 0) {
            int local_rc = checked([&] {
                return operations.retire();
            });
            int agreement_rc = checked([&] {
                return operations.agree(Phase::RETIRE, local_rc);
            });
            rc = local_rc != 0 ? local_rc : agreement_rc;
        }
        if (rc != 0) {
            try {
                operations.discard();
            } catch (...) {}
            failed_ = true;
            current.status = DomainRecoveryStatus::RECOVERY_FAILED;
            current.recovery_rc = rc;
            // A failed collective has no binding that callers may dispatch.
            current.device_ctx = 0;
            current.local_window_base = 0;
            current.workspace = 0;
            remember(request, current);
            return current;
        }
        replacement.status = DomainRecoveryStatus::REBUILT;
        replacement.generation = ++generation_;
        replacement.rebind_required = 1;
        replacement.recovery_rc = 0;
        remember(request, replacement);
        return replacement;
    }

private:
    static bool same_request(const DomainRecoveryRequest &a, const DomainRecoveryRequest &b) {
        return a.recovery_id == b.recovery_id && a.expected_generation == b.expected_generation &&
               a.failed_run_epoch == b.failed_run_epoch && a.impact == b.impact &&
               a.collective_quiescence == b.collective_quiescence && a.original_rc == b.original_rc &&
               a.reserved == b.reserved;
    }
    void remember(const DomainRecoveryRequest &request, const DomainRecoveryResult &result) {
        last_id_ = request.recovery_id;
        last_request_ = request;
        last_result_ = result;
    }
    uint64_t generation_{1};
    uint64_t last_id_{0};
    bool failed_{false};
    DomainRecoveryRequest last_request_{};
    DomainRecoveryResult last_result_{};
};
