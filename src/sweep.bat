

ECHO OFF

@REM FOR %%z IN (Maass,STDP,STSP) DO (
@REM     FOR %%y IN (0, 1.5) DO (
@REM         FOR %%x IN (0.0, 1.5) DO (
@REM             FOR %%w IN (700) DO (
@REM                 FOR %%v IN (.01, 0.05, .1) DO (
@REM                     py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir standard_hei --length 700
@REM                 )
@REM                 FOR %%u IN (0, .33, .66) DO (
@REM                     py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir standard_hei --length 700 
@REM                 )
@REM                 py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 5 2 --topology geo  --dir standard_hei --length 700
@REM                 py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 5 5 4 --topology geo  --dir standard_hei --length 700
@REM                 py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 100 1 1 --topology geo  --dir standard_hei --length 700
@REM             )
@REM         )
@REM     )
@REM )

@REM FOR %%z IN (STDP) DO (
@REM     FOR %%y IN (1.5) DO (
@REM         FOR %%x IN (1.5) DO (
@REM             FOR %%w IN (500,700,1000) DO (
@REM                 FOR %%v IN (0.05, .1, .3) DO (
@REM                     py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir dense_sweep --length 700
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )

@REM py -3.8 main.py --just_input True --length 100 --channels 40 --replicas 3 --patterns 3 --input_name "Poisson" --dir winner_sweep

FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
    FOR %%y IN (0, 1.5, 3) DO (
        FOR %%x IN (0.0, 1.5, 3) DO (
            FOR %%w IN (64) DO (
                FOR %%v IN (.2, 0.3, .4) DO (
                    py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir winner_sweep --length 100 --input_name "Poisson"
                )
                FOR %%u IN (0.0, .33, .66) DO (
                    py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir winner_sweep --length 100 --input_name "Poisson"
                )
                py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 4 4 4 --topology geo  --dir winner_sweep --length 100 --input_name "Poisson"
                py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 8 4 2 --topology geo  --dir winner_sweep --length 100 --input_name "Poisson"
                py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 16 2 2 --topology geo  --dir winner_sweep --length 100 --input_name "Poisson"
            )
        )
    )
)

@REM FOR %%i IN (1,2,3,4,5,6,7,8,9,10) DO (
@REM     py -3.8 main.py --learning Maass --topology geo --dims 8 8 1  --dir scale_testing --refractory 3 --delay 2 --input_name "Poisson" --length 100 --input_sparsity 0.02 --res_sparsity 0.01 --neurons 64 --STSP_U %%i
@REM )
PAUSE
