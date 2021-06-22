clean:
	rm -rf ./*trash*

test:
	./tests/libs/bats/bin/bats tests/test.bats

format:
	black -l 79 .

install-test-reqs:
	mkdir -p tests/libs
	git submodule add https://github.com/sstephenson/bats tests/libs/bats
