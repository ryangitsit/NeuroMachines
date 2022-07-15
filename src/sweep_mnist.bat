ECHO OFF

FOR %%i IN (.1,.2,.3) DO (
    FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
        FOR %%y IN (0) DO (
            FOR %%x IN (1.5) DO (
                FOR %%w IN (135) DO (
                    FOR %%v IN (.1,.2,.3) DO (
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --input_sparsity %%i  --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --input_sparsity %%i --x_atory "True" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --input_sparsity %%i --feed "continuous" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "False"

                        FOR %%u IN (0.0, .33, .66) DO (
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "False"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "False"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "False"
                            py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "False"
                        )

                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "False"

                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "False"

                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "False"
                        py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST_asymm --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "False"
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
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --input_sparsity %%i  --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"

@REM                         FOR %%u IN (0.0, .33, .66) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"
@REM                         )

@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 15 3 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"

@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 9 5 3 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"

@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --feed "continuous" --symmin "True"
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --topology geo --dims 27 5 1 --dir SuperSweep_MNIST --length 350 --input_name "MNIST" --res_sparsity %%v --input_sparsity %%i --x_atory "True" --feed "continuous" --symmin "True"
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )


PAUSE