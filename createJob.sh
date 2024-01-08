
#PBS -N GPUDeepAnt
#PBS -l select=1:ngpus=1

#PBS -l walltime=300:00:00 
#PBS -oe
#PBS -m abe
#PBS -M emerson.okano@unifesp.br

  
#PBS -V
python ~/shapExplain/run_YAHOO.py >> log/log_YAHOO_Faltando.log
