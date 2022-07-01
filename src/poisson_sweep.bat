ECHO OFF

FOR %%i IN (.1, .2, .3, .4) DO (
    FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
        FOR %%y IN (0) DO (
            FOR %%x IN (1.5) DO (
                FOR %%w IN (135) DO (
                    FOR %%v IN (.1,.2,.3, .4) DO (
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir poisson_final --length 500 --input_name "Poisson" --input_sparsity %%i --x_atory True
                        FOR %%u IN (.33) DO (
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir poisson_finalt --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --x_atory True
                        )
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 15 3 3 --topology geo  --dir poisson_final --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --x_atory True
                    )
                )
            )
        )
    )
)
PAUSE