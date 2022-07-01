
ECHO OFF


FOR %%i IN (.1,.2,.3) DO (
    FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
        FOR %%y IN (0) DO (
            FOR %%x IN (1.5) DO (
                FOR %%w IN (135) DO (
                    FOR %%v IN (.1,.2,.3) DO (
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep --length 700 --input_name "Heidelberg" --input_sparsity %%i
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep --length 700 --input_name "Heidelberg" --input_sparsity %%i --x_atory True
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep --length 700 --input_name "Heidelberg" --input_sparsity %%i --feed "continuous"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep --length 700 --input_name "Heidelberg" --input_sparsity %%i --x_atory True --feed "continuous"

                        FOR %%u IN (0.0, .33, .66) DO (
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True --feed "continuous"
                        )
                        FOR %%l IN (2, 4, 8) DO (
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --lamb %%l --topology geo --dims 15 3 3 --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --lamb %%l --topology geo --dims 15 3 3 --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --lamb %%l --topology geo --dims 15 3 3 --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --lamb %%l --topology geo --dims 15 3 3 --dir SuperSweep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True --feed "continuous"
                        )
                    )
                )
            )
        )
    )
)


@REM Rerun an experiment many times
FOR /L %%i IN (1,1,100) DO (
    python main.py --learning LSTP --topology geo --dims 15 3 3 --lamb 8 --neurons 135 --refractory 0 --delay 1.5 --res_sparsity 0.2 --input_sparsity 0.3 --length 700 --replicas 3 --patterns 3 --input_name "Heidelberg" --feed "reset" --x_atory False --dir rerun_LSTP --ID %%i
)

PAUSE