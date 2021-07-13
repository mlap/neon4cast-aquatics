#!/usr/bin/env bats

@test "Testing DO" {
  bash tests/test-do.sh
}

# @test "Testing WT" {
#   bash tests/test-wt.sh
# }

@test "Testing Tuning" {
  bash tests/test-tune.sh
}
