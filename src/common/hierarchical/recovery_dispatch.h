/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <type_traits>
#include "../platform_comm/domain_recovery.h"
#include "../worker/runtime_c_api.h"
#include "../task_interface/recovery_domain_binding.h"

enum class RecoveryControlPhase : uint32_t { INSPECT, PREPARE, COMMIT, REARM, ARM, ABORT };

// Local control payload; both processes use the native codec, never Python offsets.
struct RecoveryControlPacket {
    uint32_t abi_version{1};
    uint32_t struct_bytes{sizeof(RecoveryControlPacket)};
    uint64_t recovery_id{0};
    uint64_t endpoint_generation{0};
    RecoveryControlPhase phase{RecoveryControlPhase::INSPECT};
    DomainRecoveryImpact impact{DomainRecoveryImpact::EXECUTION_ONLY};
    uint32_t has_domain{0};
    uint32_t faulted{0};
    RecoveryDomainBinding before{};
    ExecutionRecoveryResult execution{};
    DomainRecoveryResult domain{};
};
static_assert(std::is_standard_layout_v<RecoveryControlPacket> && std::is_trivially_copyable_v<RecoveryControlPacket>);
