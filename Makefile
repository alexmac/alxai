all:
	uv run basedpyright --createstub alxai
	rsync -a typings/alxai/ alxai/
	uv build
	twine upload dist/*
clean:
	find alxai |  grep pyi | xargs rm
	rm -rf dist alxai.egg-info typings
check:
	uv run basedpyright --level error --threads 2 alxai tests example