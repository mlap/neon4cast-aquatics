#!/usr/bin/env bats

@test "Testing DO" {
  bash tests/test-do.sh
}

# I don't have the csv for this in the dir atm
# @test "Testing WT" {
#   bash tests/test-wt.sh
# }

@test "Testing Tuning" {
  bash tests/test-tune.sh
}
