install:
	pip install -r app/requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo "\n## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md
	echo "\n## Evidently Report" >> report.md
	echo "[Klik untuk lihat report Evidently](./Results/evidently_report.html)" >> report.md
	echo "\n## Monitoring Dashboard" >> report.md
	echo "[Klik untuk lihat Monitoring Dashboard](./Monitoring/dashboard.html)" >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update results"
	git push --force origin HEAD:update

monitor:
	python monitor.py
