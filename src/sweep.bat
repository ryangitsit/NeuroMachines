
ECHO OFF


FOR %%i IN (.1,.2,.3,.4,.5) DO (
    FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
        FOR %%y IN (0) DO (
            FOR %%x IN (1.5) DO (
                FOR %%w IN (135) DO (
                    FOR %%v IN (.1,.2,.3,.4,.5) DO (
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweepXL --length 700 --input_name "Heidelberg" --input_sparsity %%i
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweepXL --length 700 --input_name "Heidelberg" --input_sparsity %%i --x_atory "True"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweepXL --length 700 --input_name "Heidelberg" --input_sparsity %%i --feed "continuous"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweepXL --length 700 --input_name "Heidelberg" --input_sparsity %%i --x_atory "True" --feed "continuous"

                        FOR %%u IN (0.0, .33, .66) DO (
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous"
                        )

                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous"

                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous"

                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweepXL --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous"
                    )
                )
            )
        )
    )
)

@REM FOR %%i IN (.1,.2,.3) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%y IN (0) DO (
@REM             FOR %%x IN (1.5) DO (
@REM                 FOR %%w IN (135) DO (
@REM                     FOR %%v IN (.1,.2,.3) DO (
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep5 --length 700 --input_name "Heidelberg" --input_sparsity %%i  --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep5 --length 700 --input_name "Heidelberg" --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep5 --length 700 --input_name "Heidelberg" --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep5 --length 700 --input_name "Heidelberg" --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"

@REM                         FOR %%u IN (0.0, .33, .66) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"
@REM                         )

@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"

@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"

@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep5 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )



@REM @REM Rerun an experiment many times
@REM @REM  Maass_smw=(randNone_geoNone_sm0.66)_N=135_IS=0.2_RS=0.1_ref=0.0_delay=1.5_U=0.6_XTrue_feedcontinuous_IDNone
@REM FOR /L %%i IN (1,1,10) DO (
@REM     python main.py --learning Maass --topology smw --beta 0.66  --neurons 135 --refractory 0 --delay 1.5 --res_sparsity 0.1 --input_sparsity 0.2 --length 700 --replicas 3 --patterns 3 --input_name "Heidelberg" --feed "continuous" --x_atory True --dir rerun_Maass --seeding "False" --ID %%i --save_spikes 0
@REM )

@REM @REM STSP_geo=(randNone_geo2_smNone)_N=135_IS=0.2_RS=0.2_ref=0.0_delay=1.5_U=0.6_XFalse_feedcontinuous_IDNone
@REM FOR /L %%i IN (1,1,10) DO (
@REM     python main.py --learning STSP --topology geo --dims 15 3 3 --lamb 2 --neurons 135 --refractory 0 --delay 1.5 --res_sparsity 0.2 --input_sparsity 0.2 --length 700 --replicas 3 --patterns 3 --input_name "Heidelberg" --feed "continuous" --x_atory "False" --dir rerun_STSP --seeding "False" --ID %%i --save_spikes 0
@REM )

@REM @REM  STDP_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.3_RS=0.3_ref=0.0_delay=1.5_U=0.6_XTrue_feedcontinuous_IDNone
@REM FOR /L %%i IN (1,1,10) DO (
@REM     python main.py --learning STDP --topology smw --beta 0.0 --neurons 135 --refractory 0 --delay 1.5 --res_sparsity 0.3 --input_sparsity 0.3 --length 700 --replicas 3 --patterns 3 --input_name "Heidelberg" --feed "continuous" --x_atory False --dir rerun_STDP --seeding "False" --ID %%i --save_spikes 0
@REM )

@REM @REM LSTP_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.1_RS=0.3_ref=0.0_delay=1.5_U=0.6_XFalse_feedreset_IDNone
@REM FOR /L %%i IN (1,1,10) DO (
@REM     python main.py --learning LSTP --topology smw --beta 0.0 --neurons 135 --refractory 0 --delay 1.5 --res_sparsity 0.3 --input_sparsity 0.1 --length 700 --replicas 3 --patterns 3 --input_name "Heidelberg" --feed "reset" --x_atory False --dir rerun_LSTP --seeding "False" --ID %%i --save_spikes 0
@REM )



@REM Lite Sweep
@REM FOR %%i IN (.1,.2,.3) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%y IN (0) DO (
@REM             FOR %%x IN (1.5) DO (
@REM                 FOR %%w IN (135) DO (
@REM                     FOR %%v IN (.1,.2,.3) DO (
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir LightSweep_4r --length 700 --input_name "Heidelberg" --input_sparsity %%i --replicas 4

@REM                         FOR %%u IN (0.0, .33, .66) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir LightSweep_4r --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --replicas 4
@REM                         )
@REM                         FOR %%l IN (2, 4, 8) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --lamb %%l --topology geo --dims 15 3 3 --dir LightSweep_4r --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --replicas 4
@REM                         )
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )


PAUSE