/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../platform_comm/domain_recovery.h"
#include "task_args_wire.h"

struct MaterializedRecoveryBinding {
    ChipStorageTaskArgs args{};
    std::vector<TaskDomainBinding> domain_bindings;
};

// Argument ownership is explicit. Numeric values that happen to equal an
// address are never searched or rewritten. This mechanism never dispatches.
class RecoveryBindingPlan {
public:
    explicit RecoveryBindingPlan(uint64_t recovery_id) :
        recovery_id_(recovery_id) {
        if (recovery_id == 0) throw std::invalid_argument("binding plan requires a recovery id");
    }
    uint64_t recovery_id() const { return recovery_id_; }
    bool affects_domain(int32_t worker_id, uint64_t allocation_id) const {
        return find(worker_id, allocation_id) != nullptr;
    }

    void add_domain(int32_t worker_id, const RecoveryDomainBinding &before, const DomainRecoveryResult &result) {
        validate_domain(before);
        if (worker_id < 0 || result.abi_version != 2 || result.struct_bytes != sizeof(DomainRecoveryResult) ||
            result.recovery_id != recovery_id_ || result.allocation_id != before.allocation_id ||
            result.previous_generation != before.generation || result.recovery_rc != 0 ||
            result.domain_preserved != 1 ||
            (result.status != DomainRecoveryStatus::PRESERVED && result.status != DomainRecoveryStatus::REARMED &&
             result.status != DomainRecoveryStatus::ALREADY_RECOVERED))
            throw std::invalid_argument("binding plan requires a matching committed domain result");
        RecoveryDomainBinding after{result.allocation_id,     result.generation,  result.device_ctx,
                                    result.local_window_base, result.window_size, result.workspace};
        validate_domain(after);
        if (after.generation == before.generation) {
            if (!(after == before) || result.status == DomainRecoveryStatus::REARMED || result.rebind_required != 0)
                throw std::invalid_argument("preserved binding must retain its generation and resources");
        } else if (before.generation == UINT64_MAX || after.generation != before.generation + 1 ||
                   result.status == DomainRecoveryStatus::PRESERVED || result.rebind_required != 1) {
            throw std::invalid_argument("rearmed binding must advance exactly one generation");
        }
        if (after.generation != before.generation &&
            (result.transport_stopped != 1 || result.transport_channel_count == 0 ||
             result.stopped_channel_count != result.transport_channel_count))
            throw std::invalid_argument("rearmed binding lacks complete transport stop proof");
        if (find(worker_id, before.allocation_id) != nullptr)
            throw std::invalid_argument("duplicate worker/domain in binding plan");
        domains_.push_back(Change{worker_id, before, after});
    }

    // The owner registers replacement backings before supplying their descriptors.
    // A changed wire address cannot retain its import-cache canonical identity.
    void add_tensor_replacement(int32_t worker_id, uint64_t allocation_id, uint32_t index, const Tensor &tensor) {
        validate_tensor(tensor);
        if (find(worker_id, allocation_id) == nullptr) throw std::invalid_argument("unknown replacement domain");
        for (const auto &replacement : replacements_)
            if (replacement.worker_id == worker_id && replacement.allocation_id == allocation_id &&
                replacement.index == index)
                throw std::invalid_argument("duplicate tensor replacement");
        replacements_.push_back(Replacement{worker_id, allocation_id, index, tensor});
    }

    TaskArgs rebind(const TaskArgs &source, int32_t worker_id) const {
        TaskArgs next = source;
        rewrite(next, next.domain_bindings, worker_id);
        return next;
    }

    // Historical plans may precede an argument's generation. The latest plan
    // still validates its final resources; a fabricated future binding fails.
    TaskArgs advance(const TaskArgs &source, int32_t worker_id) const {
        TaskArgs next = source;
        for (auto &binding : next.domain_bindings) {
            const auto *change = find(worker_id, binding.domain.allocation_id);
            if (change == nullptr || binding.domain.generation > change->after.generation) continue;
            std::vector<TaskDomainBinding> selected{binding};
            rewrite(next, selected, worker_id);
            binding = std::move(selected.front());
        }
        return next;
    }

    void validate_current(const TaskArgs &args, int32_t worker_id) const {
        for (const auto &binding : args.domain_bindings) {
            const auto *change = find(worker_id, binding.domain.allocation_id);
            if (change == nullptr || !(binding.domain == change->after))
                throw std::invalid_argument("submission does not match the latest committed binding");
            validate_binding(args, binding);
        }
    }

    MaterializedRecoveryBinding rebind_materialized(
        const ChipStorageTaskArgs &source, const std::vector<TaskDomainBinding> &bindings, int32_t worker_id
    ) const {
        MaterializedRecoveryBinding next{source, bindings};
        rewrite(next.args, next.domain_bindings, worker_id);
        return next;
    }

    void validate_workers(const std::vector<int32_t> &workers) const {
        if (domains_.empty()) throw std::invalid_argument("empty binding plan");
        for (const auto &domain : domains_)
            if (std::find(workers.begin(), workers.end(), domain.worker_id) == workers.end())
                throw std::invalid_argument("binding plan targets a different worker");
    }

    static void validate_binding(const TaskArgs &args, const TaskDomainBinding &binding) {
        validate_domain(binding.domain);
        if (binding.arguments.empty()) throw std::invalid_argument("domain binding has no arguments");
        std::set<std::pair<bool, uint32_t>> indices;
        for (const auto &argument : binding.arguments) {
            if (!indices.emplace(argument.kind == RecoveryArgKind::WINDOW_TENSOR, argument.index).second)
                throw std::invalid_argument("duplicate argument binding");
            validate_argument(args, argument, binding.domain);
        }
    }

private:
    struct Change {
        int32_t worker_id;
        RecoveryDomainBinding before;
        RecoveryDomainBinding after;
    };
    struct Replacement {
        int32_t worker_id;
        uint64_t allocation_id;
        uint32_t index;
        Tensor tensor;
    };
    uint64_t recovery_id_;
    std::vector<Change> domains_;
    std::vector<Replacement> replacements_;

    const Change *find(int32_t worker_id, uint64_t allocation_id) const {
        for (const auto &domain : domains_)
            if (domain.worker_id == worker_id && domain.before.allocation_id == allocation_id) return &domain;
        return nullptr;
    }
    static void validate_domain(const RecoveryDomainBinding &domain) {
        if (domain.generation == 0 || domain.device_ctx == 0 || domain.local_window_base == 0 ||
            domain.window_size == 0 || domain.window_size > UINT64_MAX - domain.local_window_base)
            throw std::invalid_argument("invalid recovery domain binding");
    }
    static uint64_t address(const RecoveryDomainBinding &domain, uint64_t offset, uint64_t bytes = 1) {
        if (offset >= domain.window_size || bytes > domain.window_size - offset)
            throw std::invalid_argument("argument lies outside its domain window");
        return domain.local_window_base + offset;
    }
    static uint64_t tensor_address(const Tensor &tensor) {
        validate_tensor(tensor);
        if (tensor.buffer.backend_kind != static_cast<uint8_t>(BackendKind::VMM_WINDOW) ||
            tensor.buffer.address_space != static_cast<uint8_t>(AddressSpace::DEVICE))
            throw std::invalid_argument("wire tensor binding requires a registered device window");
        uint64_t base = 0;
        std::memcpy(&base, tensor.buffer.body, sizeof(base));
        return base;
    }
    static uint64_t tensor_address(const ChipTensor &tensor) {
        if (!tensor.is_device_memory() || tensor.buffer.size == 0 || tensor.ndims == 0 ||
            tensor.ndims > MAX_TENSOR_DIMS || tensor.dtype >= DataType::DATA_TYPE_NUM)
            throw std::invalid_argument("materialized tensor binding requires device storage");
        const uint64_t element_bytes = get_element_size(tensor.dtype);
        if (element_bytes == 0) throw std::invalid_argument("invalid materialized tensor dtype");
        const uint64_t capacity = tensor.buffer.size / element_bytes;
        if (tensor.start_offset >= capacity)
            throw std::invalid_argument("materialized tensor view exceeds its backing");
        uint64_t last = tensor.start_offset;
        for (uint32_t i = 0; i < tensor.ndims; ++i) {
            if (tensor.shapes[i] == 0 || tensor.strides[i] == 0)
                throw std::invalid_argument("invalid materialized tensor shape or stride");
            const uint64_t span = static_cast<uint64_t>(tensor.shapes[i] - 1) * tensor.strides[i];
            if (span >= capacity - last) throw std::invalid_argument("materialized tensor view exceeds its backing");
            last += span;
        }
        return tensor.buffer.addr;
    }
    static uint64_t tensor_bytes(const Tensor &tensor) { return tensor.buffer.nbytes; }
    static uint64_t tensor_bytes(const ChipTensor &tensor) { return tensor.buffer.size; }

    template <typename Args>
    static void
    validate_argument(const Args &args, const RecoveryArgBinding &argument, const RecoveryDomainBinding &domain) {
        if (argument.kind == RecoveryArgKind::WINDOW_TENSOR) {
            if (argument.index >= static_cast<uint32_t>(args.tensor_count()))
                throw std::out_of_range("binding tensor index");
            const auto &tensor = args.tensor(static_cast<int32_t>(argument.index));
            if (tensor_address(tensor) != address(domain, argument.offset, tensor_bytes(tensor)))
                throw std::invalid_argument("binding does not match the tensor backing");
        } else {
            if (argument.index >= static_cast<uint32_t>(args.scalar_count()))
                throw std::out_of_range("binding scalar index");
            uint64_t expected = 0;
            if (argument.kind == RecoveryArgKind::COMM_CONTEXT && argument.offset == 0) expected = domain.device_ctx;
            else if (argument.kind == RecoveryArgKind::WINDOW_ADDRESS) expected = address(domain, argument.offset);
            else throw std::invalid_argument("invalid argument binding kind or offset");
            if (args.scalar(static_cast<int32_t>(argument.index)) != expected)
                throw std::invalid_argument("binding does not match the scalar address");
        }
    }

    void update_tensor(
        ChipTensor &tensor, const RecoveryDomainBinding &after, const RecoveryArgBinding &argument, int32_t, uint64_t
    ) const {
        tensor.buffer.addr = address(after, argument.offset, tensor.buffer.size);
    }
    void update_tensor(
        Tensor &tensor, const RecoveryDomainBinding &after, const RecoveryArgBinding &argument, int32_t worker_id,
        uint64_t allocation_id
    ) const {
        const uint64_t target = address(after, argument.offset, tensor.buffer.nbytes);
        const Replacement *replacement = nullptr;
        for (const auto &candidate : replacements_)
            if (candidate.worker_id == worker_id && candidate.allocation_id == allocation_id &&
                candidate.index == argument.index)
                replacement = &candidate;
        if (replacement == nullptr) {
            if (tensor_address(tensor) == target) return;
            throw std::invalid_argument("changed wire window requires a registered replacement");
        }
        const Tensor &next = replacement->tensor;
        if (tensor_address(next) != target || next.buffer.identity == tensor.buffer.identity ||
            next.buffer.nbytes != tensor.buffer.nbytes || next.buffer.access != tensor.buffer.access ||
            next.byte_offset != tensor.byte_offset || next.ndims != tensor.ndims || next.dtype != tensor.dtype ||
            std::memcmp(next.shapes, tensor.shapes, sizeof(tensor.shapes)) != 0 ||
            std::memcmp(next.strides, tensor.strides, sizeof(tensor.strides)) != 0)
            throw std::invalid_argument("replacement changes tensor view, access or import-cache identity");
        tensor = next;
    }

    template <typename Args>
    void rewrite(Args &args, std::vector<TaskDomainBinding> &bindings, int32_t worker_id) const {
        if (worker_id < 0 || bindings.empty()) throw std::invalid_argument("task has no explicit domain bindings");
        std::set<uint64_t> domain_ids;
        std::set<std::pair<bool, uint32_t>> indices;
        for (auto &binding : bindings) {
            validate_domain(binding.domain);
            if (binding.arguments.empty() || !domain_ids.insert(binding.domain.allocation_id).second)
                throw std::invalid_argument("empty or duplicate task domain binding");
            const Change *change = find(worker_id, binding.domain.allocation_id);
            if (change == nullptr) throw std::invalid_argument("binding plan is missing a task domain");
            const bool already_bound = binding.domain == change->after;
            if (!already_bound && !(binding.domain == change->before))
                throw std::invalid_argument("stale task domain generation or resources");
            for (const auto &argument : binding.arguments) {
                if (!indices.emplace(argument.kind == RecoveryArgKind::WINDOW_TENSOR, argument.index).second)
                    throw std::invalid_argument("argument belongs to multiple bindings");
                validate_argument(args, argument, binding.domain);
                if (already_bound) continue;
                if (argument.kind == RecoveryArgKind::WINDOW_TENSOR)
                    update_tensor(
                        args.tensor(static_cast<int32_t>(argument.index)), change->after, argument, worker_id,
                        binding.domain.allocation_id
                    );
                else
                    args.scalar(static_cast<int32_t>(argument.index)) = argument.kind == RecoveryArgKind::COMM_CONTEXT ?
                                                                            change->after.device_ctx :
                                                                            address(change->after, argument.offset);
            }
            binding.domain = change->after;
        }
    }
};
