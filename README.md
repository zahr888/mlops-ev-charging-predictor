# Quick Start

- **Docker:** Start local services used by the project (e.g. LocalStack) with Docker Compose.

	- From the repository root (PowerShell):

		```powershell
		docker compose -f .\docker-compose.yml up -d
		```

**Summary of Recent Changes**

- **Notebook fix:** Updated the plotting cell in `notebooks/01_eda.ipynb` so month values are coerced to numeric (1–12), missing rows are dropped before plotting, and x-axis ticks are set to `1..12`. This ensures months appear in chronological order on the scatter plots.
- **Validation:** The notebook JSON was validated after the edit to avoid corruption.

**Useful Commands / Examples**

- Run the ingest helper to convert a CSV to Parquet and optionally upload to S3 (LocalStack):

	```powershell
	python .\src\ingest\save_parquet.py --csv "./data/downloaded/Dataset 2_Hourly EV loads - Per user.csv" --output .\data\raw --upload
	```

- Open and run the exploratory notebook (Jupyter Lab):

	```powershell
	jupyter lab
	# then open `notebooks/01_eda.ipynb` in the browser and run the cells
	```

- Execute the notebook headlessly (useful for CI or automation):

	```powershell
	jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
	```

**Notes & Tips**

- If `month_plugin` is stored as a string (e.g. `'Jan'` or `'1'`), the notebook code will coerce it to numeric months; if you prefer month names on the x-axis I can update the cell to show `Jan..Dec` labels instead.
- The ingest script uses LocalStack by default in the repo for S3-compatible testing — ensure the local stack is running (via Docker Compose) before using `--upload`.

**Next steps (optional)**

- Add a small README section describing how to run unit tests or the model training pipeline.
- Add month-name tick labels or an aggregated monthly summary plot in `01_eda.ipynb`.

If you want, I can also commit the notebook changes to a branch and push, or add month-name labels and a small aggregation overlay (median/mean per month).


