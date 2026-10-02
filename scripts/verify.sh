#!/usr/bin/env sh
# Local verification before every push (private repo: CI does not run on push).
set -eu
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo check -p typedlm --no-default-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
# MSRV 1.85 for everything but `optimize`, whose dependency pathwise needs 1.88.
cargo +1.85 check --workspace --all-targets --features typedlm/derive,typedlm/http,typedlm/eval
cargo +1.88.0 check -p typedlm --all-targets --all-features
