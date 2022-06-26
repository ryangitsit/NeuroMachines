
ECHO OFF

py -3.8 main.py --just_input True --length 500 --channels 40 --replicas 3 --patterns 3 --input_name "Poisson" --dir poisson_no-reset
FOR %%i IN (.18, .2, .22) DO (
    FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
        FOR %%y IN (0) DO (
            FOR %%x IN (1.5) DO (
                FOR %%w IN (100) DO (
                    FOR %%v IN (.1,.2,.3) DO (
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir poisson_no-reset --length 500 --input_name "Poisson" --input_sparsity %%i
                        FOR %%u IN (.33) DO (
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir poisson_no-reset --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
                        )
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 5 2 --topology geo  --dir poisson_no-reset --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
                    )
                )
            )
        )
    )
)

PAUSE