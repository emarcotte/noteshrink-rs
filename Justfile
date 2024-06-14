sample:
	cargo run -- ./samples/notesA1.jpg

test-watch:
	cargo watch -x 'llvm-cov test --lcov --output-path coverage.lcov'