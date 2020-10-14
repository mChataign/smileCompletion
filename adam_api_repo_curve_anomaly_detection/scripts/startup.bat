SET CDIR=%~dp0

SET ROOTDIR=%CDIR%..
CD %ROOTDIR%

SET PROXY="http://efx-nexus.systems.uk.hsbc:8083/nexus/repository/pypi.proxy/simple"
SET extra_index_url="http://efx-nexus.systems.uk.hsbc:8083/nexus/repository/pypi.hosted/simple"

SET REQUIREMENTS=%ROOTDIR%\requirements.txt
SET TMP_ENV=%ROOTDIR%\venv

python -m pip install virtualenv

IF NOT EXIST %TMP_ENV% (
    python -m virtualenv %TMP_ENV%
    python -m pip install --ignore-installed --index-url %PROXY% -r %REQUIREMENTS%
    python -m pip install --ignore-installed --index-url %PROXY% --extra-index-url %extra_index_url% -r %REQUIREMENTS%
) ELSE (
    python -m pip install --index-url %PROXY% -r %REQUIREMENTS%
    python -m pip install --index-url %PROXY% --extra-index-url %extra_index_url% -r %REQUIREMENTS%
)

START "" /B /D . python %ROOTDIR%\main.py