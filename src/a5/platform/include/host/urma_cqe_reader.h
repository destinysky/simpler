/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <cstdint>
#include <functional>

#include "platform_comm/domain_recovery.h"

using UrmaCqeReader = std::function<int(uint64_t, uint32_t &, uint32_t)>;

int comm_recover_domain_with_reader(
    void *handle, uint64_t allocation_id, const DomainRecoveryRequest *request, DomainRecoveryResult *result,
    const UrmaCqeReader &read_cqe
);
