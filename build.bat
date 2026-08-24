@echo off
chcp 65001 >nul
REM =====================================================================
REM  EnergyLevel 一键打包（onedir 首发版）
REM  先决条件：本项目根已存在构建环境 build_venv，创建方法：
REM      D:\python310\python.exe -m venv build_venv
REM  用法：双击 build.bat（或在命令行运行 build.bat）
REM  产物：dist\EnergyLevel\EnergyLevel.exe（连同整个文件夹一起分发）
REM =====================================================================
cd /d "%~dp0"

if not exist "build_venv\Scripts\python.exe" (
    echo [错误] 未找到 build_venv。请先执行：
    echo   D:\python310\python.exe -m venv build_venv
    pause
    exit /b 1
)

echo [1/3] 安装/同步依赖（清华镜像，已装则秒过）...
build_venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt -r requirements-build.txt
if errorlevel 1 ( echo [错误] 依赖安装失败 & pause & exit /b 1 )

echo [2/3] PyInstaller 打包（onedir）...
build_venv\Scripts\pyinstaller.exe --noconfirm --clean EnergyLevel.spec
if errorlevel 1 ( echo [错误] 打包失败，见上方日志 & pause & exit /b 1 )

echo [3/3] 完成！
echo   产物：dist\EnergyLevel\EnergyLevel.exe
echo   分发：把整个 dist\EnergyLevel 文件夹压成 zip 即可
echo        （首启会把 .arcdata 种子复制到 %%LOCALAPPDATA%%\EnergyLevel\.arcdata）
pause