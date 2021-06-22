clean:
	rm -rf ./*trash*

test:
	./tests/libs/bats/bin/bats tests/test.bats

format:
	black -l 79 .

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 . --count --exit-zero --statistics

install-test-reqs:
	mkdir -p tests/libs
	git submodule add --force https://github.com/sstephenson/bats tests/libs/bats
