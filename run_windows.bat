@echo off

SET DATASET=%1
IF "%DATASET%"=="" SET DATASET=kiba

SET EPOCHS=%2
IF "%EPOCHS%"=="" SET EPOCHS=100

SET BATCH_SIZE=%3
IF "%BATCH_SIZE%"=="" SET BATCH_SIZE=64

SET LR=%4
IF "%LR%"=="" SET LR=0.001

SET SMI_ENCODER=%5
IF "%SMI_ENCODER%"=="" SET SMI_ENCODER=chemberta

SET EXTRA_ARGS=%6 %7 %8 %9

echo ==================================================
echo    PhyNet-DTA 5-Fold Cross Validation (Windows)
echo    Dataset:    %DATASET%
echo    Epochs:     %EPOCHS%
echo    Batch Size: %BATCH_SIZE%
echo    Learning R: %LR%
echo    Encoder:    %SMI_ENCODER%
echo    Extra Args: %EXTRA_ARGS%
echo ==================================================
echo Starting 5-Fold Cross Validation...

:: Use explicit python executable to avoid environment issues
FOR /L %%i IN (0,1,4) DO (
    echo.
    echo ===================================
    echo Running Fold %%i...
    echo ===================================
    .\.venv\Scripts\python.exe run.py --dataset %DATASET% --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --smi_encoder %SMI_ENCODER% --val_fold %%i %EXTRA_ARGS%
)

echo.
echo All folds completed! Results saved to results/ directory.
