all: stubs
	uv build
	twine upload dist/*
stubs:
	uv run basedpyright --createstub alxai
	uv run basedpyright --createstub alxai.base
	uv run basedpyright --createstub alxai.openai
	uv run basedpyright --createstub alxai.anthropic
	rsync -a typings/alxai/ alxai/
clean:
	find alxai |  grep pyi | xargs rm
	rm -rf dist alxai.egg-info typings
check:
	uv run basedpyright --level error --threads 2 alxai tests example