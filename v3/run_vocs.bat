@echo off
echo ========================================
echo Weekly VoCS Analysis Starting
echo ========================================
echo Time: %date% %time%
echo.

cd /d "C:\Users\DimosPapadopoulos\OneDrive - ATG Entertainment\work\CES_VoiceSynthesizer_Agent"

echo Loading environment variables from .env file...
echo NOTE: In production, set these as environment variables or use .env file


ANTHROPIC_API_KEY=
SLACK_WEBHOOK_CHECKOUT=
SLACK_WEBHOOK_CATALOGUE=
SLACK_WEBHOOK_ACCOUNT=
SLACK_WEBHOOK_DAS=

echo.
echo Running VoCS analysis...
py vocs_multi_team.py

echo.
echo ========================================
echo VoCS Analysis Complete
echo ========================================
echo Time: %date% %time%
pause