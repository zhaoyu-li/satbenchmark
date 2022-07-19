# easy
python generators/sr.py ~/scratch/satbenchmark/easy/sr/train 80000 --min_n 10 --max_n 200
python generators/sr.py ~/scratch/satbenchmark/easy/sr/valid 10000 --min_n 10 --max_n 200
python generators/sr.py ~/scratch/satbenchmark/easy/sr/test 10000 --min_n 10 --max_n 200

python generators/3-sat.py ~/scratch/satbenchmark/easy/3-sat/train 80000 --min_n 10 --max_n 100
python generators/3-sat.py ~/scratch/satbenchmark/easy/3-sat/valid 10000 --min_n 10 --max_n 100
python generators/3-sat.py ~/scratch/satbenchmark/easy/3-sat/test 10000 --min_n 10 --max_n 100

# medium
python generators/sr.py ~/scratch/satbenchmark/medium/sr/train 80000 --min_n 200 --max_n 400
python generators/sr.py ~/scratch/satbenchmark/medium/sr/valid 10000 --min_n 200 --max_n 400
python generators/sr.py ~/scratch/satbenchmark/medium/sr/test 10000 --min_n 200 --max_n 400

python generators/3-sat.py ~/scratch/satbenchmark/medium/3-sat/train 80000 --min_n 100 --max_n 200
python generators/3-sat.py ~/scratch/satbenchmark/medium/3-sat/valid 10000 --min_n 100 --max_n 200
python generators/3-sat.py ~/scratch/satbenchmark/medium/3-sat/test 10000 --min_n 100 --max_n 200

# hard
python generators/sr.py ~/scratch/satbenchmark/hard/sr/test 10000 --min_n 400 --max_n 800

python generators/3-sat.py ~/scratch/satbenchmark/hard/3-sat/test 10000 --min_n 200 --max_n 400