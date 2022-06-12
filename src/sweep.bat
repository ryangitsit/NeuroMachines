ECHO OFF

py -3.8 ./just_input.py --location full_sweep

FOR %%z IN (Maass,STDP,STSP) DO (
    FOR %%y IN (0, 1.5) DO (
        FOR %%x IN (0.0, 1.5, 3.0) DO (
            FOR %%w IN (64) DO (
                FOR %%v IN (.15, 0.3, .45) DO (
                    py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd --new_input False --location hei_sweep --length 700
                )
                FOR %%u IN (0, .33, .66) DO (
                    py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --new_input False --location hei_sweep --length 700
                )
                py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 4 4 4 --topology geo --new_input False --location hei_sweep --length 700
                py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 16 2 2 --topology geo --new_input False --location hei_sweep --length 700
                py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 8 8 1 --topology geo --new_input False --location hei_sweep --length 700
            )
        )
    )
)


PAUSE

@REM For Maass, redo with only 0 refract 2.0 delay, and then everything else