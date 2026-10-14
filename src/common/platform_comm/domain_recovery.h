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
    DOMAIN_TRANSPORT = 2,
    SHARED_COMMUNICATION = 3,
    DEVICE = 4,
};

enum class DomainRecoveryPhase : uint32_t { PREPARE = 0, COMMIT = 1, ABORT = 2 };
enum class DomainRecoveryEngine : uint32_t { NONE = 0, SDMA = 1, URMA = 2, CCU = 3 };

enum class DomainRecoveryStatus : uint32_t {
    PRESERVED = 0,
    ALREADY_RECOVERED = 2,
    INVALID_STATE = 3,
    STALE_GENERATION = 4,
    BUSY = 5,
    QUIESCENCE_UNPROVEN = 6,
    UNSUPPORTED = 7,
    GIVE_UP = 8,
    RECOVERY_FAILED = 9,
    REARMED = 10,
    PREPARED = 11,
};

// The owner fences all producers on every member before PREPARE. Local
// preparation proves submitted transport work stopped and rearms resources.
// COMMIT asserts that every member prepared this transaction; ABORT supplies
// its failing member's recovery_rc. Neither phase dispatches business work.
// A diagnostic snapshot or local FIFO drain is not a collective stop proof.
struct DomainRecoveryRequest {
    uint32_t abi_version{3};
    uint32_t struct_bytes{sizeof(DomainRecoveryRequest)};
    uint64_t recovery_id{0};
    uint64_t expected_generation{0};
    uint64_t failed_run_epoch{0};
    DomainRecoveryImpact impact{DomainRecoveryImpact::EXECUTION_ONLY};
    uint32_t collective_quiescence{0};
    int32_t original_rc{0};
    DomainRecoveryPhase phase{DomainRecoveryPhase::PREPARE};
    int32_t collective_rc{0};
    uint32_t reserved{0};
};

struct DomainRecoveryResult {
    uint32_t abi_version{2};
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
    // Provider-owned transport channels: SDMA streams, or a future backend equivalent.
    uint32_t transport_channel_count{0};
    uint32_t stopped_channel_count{0};
    uint32_t transport_stopped{0};
    uint32_t reserved{0};
};

static_assert(std::is_standard_layout_v<DomainRecoveryRequest> && std::is_trivially_copyable_v<DomainRecoveryRequest>);
static_assert(std::is_standard_layout_v<DomainRecoveryResult> && std::is_trivially_copyable_v<DomainRecoveryResult>);
static_assert(sizeof(DomainRecoveryRequest) == 56 && offsetof(DomainRecoveryRequest, impact) == 32);
static_assert(sizeof(DomainRecoveryResult) == 136 && offsetof(DomainRecoveryResult, device_ctx) == 80);

// Optional host-runtime symbol. Unsupported backends keep their existing ABI.
extern "C" int comm_recover_domain(
    void *handle, uint64_t allocation_id, const DomainRecoveryRequest *request, DomainRecoveryResult *result
);

class DomainRecoveryManager {
public:
    struct Operations {
        bool preserve_supported{true};
        bool rearm_supported{false};
        std::function<int(DomainRecoveryResult &)> quiesce;
        std::function<int()> rearm;
    };

    uint64_t generation() const { return generation_; }
    bool prepared() const { return prepared_; }
    void quarantine(int rc) {
        failed_ = true;
        prepared_ = false;
        last_result_.status = DomainRecoveryStatus::RECOVERY_FAILED;
        last_result_.recovery_rc = rc;
        last_result_.domain_preserved = 0;
        last_result_.rebind_required = 0;
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
        auto reject = [&](DomainRecoveryStatus status) {
            current.status = status;
            return current;
        };
        if (request.abi_version != 3 || request.struct_bytes != sizeof(DomainRecoveryRequest))
            return reject(DomainRecoveryStatus::UNSUPPORTED);
        if (request.recovery_id == 0 || request.reserved != 0 || request.collective_quiescence > 1 ||
            (request.phase != DomainRecoveryPhase::PREPARE && request.phase != DomainRecoveryPhase::COMMIT &&
             request.phase != DomainRecoveryPhase::ABORT) ||
            (request.phase == DomainRecoveryPhase::ABORT ? request.collective_rc == 0 : request.collective_rc != 0))
            return reject(DomainRecoveryStatus::INVALID_STATE);

        if (request.recovery_id == last_id_ && same_transaction(request, last_request_)) {
            if (request.phase == DomainRecoveryPhase::ABORT) {
                // A peer may fail COMMIT after this member committed. Admission
                // stays fenced until the owner acknowledges every COMMIT.
                if (last_result_.status != DomainRecoveryStatus::PREPARED &&
                    last_result_.status != DomainRecoveryStatus::REARMED && !failed_)
                    return reject(DomainRecoveryStatus::INVALID_STATE);
                if (!failed_ || last_result_.recovery_rc == 0) quarantine(request.collective_rc);
                return last_result_;
            }
            if (request.phase == DomainRecoveryPhase::COMMIT && prepared_) {
                prepared_ = false;
                last_result_.status = DomainRecoveryStatus::REARMED;
                last_result_.generation = ++generation_;
                last_result_.domain_preserved = 1;
                last_result_.rebind_required = 1;
                return last_result_;
            }
            if (request.phase == DomainRecoveryPhase::COMMIT && last_result_.status == DomainRecoveryStatus::PRESERVED)
                return reject(DomainRecoveryStatus::INVALID_STATE);
            auto result = last_result_;
            if (result.status == DomainRecoveryStatus::PRESERVED || result.status == DomainRecoveryStatus::REARMED)
                result.status = DomainRecoveryStatus::ALREADY_RECOVERED;
            return result;
        }
        if (request.expected_generation != generation_ || request.recovery_id <= last_id_)
            return reject(DomainRecoveryStatus::STALE_GENERATION);
        if (failed_) return reject(DomainRecoveryStatus::GIVE_UP);
        if (prepared_) return reject(DomainRecoveryStatus::BUSY);
        if (request.phase != DomainRecoveryPhase::PREPARE) return reject(DomainRecoveryStatus::INVALID_STATE);
        if (request.impact == DomainRecoveryImpact::DEVICE ||
            request.impact == DomainRecoveryImpact::SHARED_COMMUNICATION ||
            (request.impact == DomainRecoveryImpact::DOMAIN_TRANSPORT && !operations.rearm_supported)) {
            current.status = DomainRecoveryStatus::GIVE_UP;
            remember(request, current);
            quarantine(0);
            last_result_.status = DomainRecoveryStatus::GIVE_UP;
            return last_result_;
        }
        if (request.impact == DomainRecoveryImpact::EXECUTION_ONLY) {
            if (!operations.preserve_supported) return reject(DomainRecoveryStatus::UNSUPPORTED);
            current.status = DomainRecoveryStatus::PRESERVED;
            current.domain_preserved = 1;
            remember(request, current);
            return current;
        }
        if (request.impact != DomainRecoveryImpact::DOMAIN_TRANSPORT)
            return reject(DomainRecoveryStatus::INVALID_STATE);
        if (!request.collective_quiescence) return reject(DomainRecoveryStatus::QUIESCENCE_UNPROVEN);
        if (generation_ == std::numeric_limits<uint64_t>::max()) return reject(DomainRecoveryStatus::GIVE_UP);
        if (!operations.quiesce || !operations.rearm) return reject(DomainRecoveryStatus::UNSUPPORTED);
        int rc = -1;
        try {
            rc = operations.quiesce(current);
            if (rc == 0 && (current.transport_stopped != 1 || current.transport_channel_count == 0 ||
                            current.stopped_channel_count != current.transport_channel_count))
                rc = -1;
            if (rc == 0) rc = operations.rearm();
        } catch (...) {
            rc = -1;
        }
        current.status = rc == 0 ? DomainRecoveryStatus::PREPARED : DomainRecoveryStatus::RECOVERY_FAILED;
        current.recovery_rc = rc;
        remember(request, current);
        if (rc != 0) {
            quarantine(rc);
        } else {
            prepared_ = true;
        }
        return last_result_;
    }

private:
    static bool same_transaction(const DomainRecoveryRequest &a, const DomainRecoveryRequest &b) {
        return a.recovery_id == b.recovery_id && a.expected_generation == b.expected_generation &&
               a.failed_run_epoch == b.failed_run_epoch && a.impact == b.impact &&
               a.collective_quiescence == b.collective_quiescence && a.original_rc == b.original_rc;
    }
    void remember(const DomainRecoveryRequest &request, const DomainRecoveryResult &result) {
        last_id_ = request.recovery_id;
        last_request_ = request;
        last_result_ = result;
    }
    uint64_t generation_{1};
    uint64_t last_id_{0};
    bool prepared_{false};
    bool failed_{false};
    DomainRecoveryRequest last_request_{};
    DomainRecoveryResult last_result_{};
};
