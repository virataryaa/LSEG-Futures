@echo off
set LOG="C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Futures\Automator\run_log.txt"
echo. >> %LOG%
echo ============================= >> %LOG%
echo Run started: %date% %time% >> %LOG%
echo ============================= >> %LOG%
python "C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Interim_Migration\Futures\Automator\run_updater.py" >> %LOG% 2>&1
echo Run finished: %date% %time% >> %LOG%
