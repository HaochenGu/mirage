# MegaEagle MPK - Customized Operations Status

## Overview
This document tracks the implementation status of all customized operations added to MPK for EAGLE integration.

## Implemented Operations

### 1. Multi-Token Embedding (TASK_MULTI_TOKEN_EMBEDDING = 114)
- **Status**: Implemented and Tested
- **Design**: One thread per token, maximizing parallelism, support single block
- **Performance**: Not optimized

### 2. Multi-Token Softmax (TASK_MULTI_TOKEN_SOFTMAX = 115)
- **Status**: Implemented and Tested 
- **Design**: One block per token
- **Performance**: Not optimized, single execution slower than pytorch

### 3. Multi-Token Linear (TASK_MULTI_TOKEN_LINEAR = 116)
- **Status**: Implemented and Tested
- **Design**: One block per token
- **Performance**: Much slower than pytorch, error is at 2e-2 level, core operation needs rewrite.


### 4. Tree Attention (TASK_TREE_ATTENTION = 117)
- **Status**: Implemented and Tested
- **Design**: Only work for single block
- **Performance**: Waiting for careful optimization
