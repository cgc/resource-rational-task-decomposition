TEST_PATH=rrtd

test:
	PYTHONPATH=rrtd py.test -m "not integrationtest" $(TEST_PATH) $(PYTEST_ARGS)

integration-test:
	PYTHONPATH=rrtd py.test -m integrationtest $(TEST_PATH) $(PYTEST_ARGS)
