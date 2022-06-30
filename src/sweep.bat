

ECHO OFF


@REM py -3.8 main.py --just_input True --length 500 --channels 40 --replicas 9 --patterns 3 --input_name "Poisson" --dir poisson_Large
@REM FOR %%i IN (.2,.3,.4,.5) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%y IN (0, 1.5, 3) DO (
@REM             FOR %%x IN (0.0, 1.5, 3) DO (
@REM                 FOR %%w IN (100) DO (
@REM                     FOR %%v IN ( .2,.3,.4,.5) DO (
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir poisson_large --length 500 --input_name "Poisson" --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir poisson_large --length 500 --input_name "Poisson" --input_sparsity %%i --x_atory True
@REM                         FOR %%u IN (0.0, .33, .66) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM                         )
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 10 1 --topology geo  --dir poisson_large  --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 5 2 --topology geo  --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 5 5 4 --topology geo  --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 10 1 --topology geo  --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 5 2 --topology geo  --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 5 5 4 --topology geo  --dir poisson_large --length 500 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )

FOR %%i IN (.1,.3,.4,.5) DO (
    FOR %%z IN (Maass,STSP,STDP,LSTP) DO (
        FOR %%y IN (0,3) DO (
            FOR %%x IN (0,1.5) DO (
                FOR %%w IN (135) DO (
                    FOR %%v IN (.2,.3,.4,.5) DO (
                        py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v --dir hei_large2_nochunkX --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3
                        py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --topology geo  --dir hei_large2_nochunkX --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3
                        py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --topology smw --dir hei_large2_nochunkX --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3
                        py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v --dir hei_large2_nochunkX --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3 --x_atory True
                        py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --topology geo  --dir hei_large2_nochunkX --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3 --x_atory True
                        py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --topology smw --dir hei_large2_nochunkX --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3 --x_atory True
                    )
                )
            )
        )
    )
)

@REM py -3.8 main.py --just_input True --length 100 --channels 800 --replicas 27 --patterns 3 --input_name "Heidelberg" --dir hei_rep
@REM FOR %%i IN (.2,.3,.4) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%w IN (1000) DO (
@REM             FOR %%v IN (.2,.3,.4) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v  --dir hei_rep --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --chunk 1 --replicas 27 --patterns 3
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v  --dir hei_rep --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --chunk 1 --x_atory True --replicas 27 --patterns 3
@REM             )
@REM         )
@REM     )
@REM )





@REM FOR %%i IN (1,2,3,4,5) DO (
@REM     py -3.8 main.py --learning STSP --topology geo --dims 4 4 4 --patterns 2 --replicas 3  --dir instant_poisson --refractory 0 --delay 0 --input_name "Poisson" --length 100 --input_sparsity 0.4 --res_sparsity 0.2 --neurons 64 --x_atory True --STSP_U %%i
@REM )




@REM py -3.8 main.py --just_input True --length 100 --channels 700 --replicas 9 --patterns 3 --input_name "Heidelberg" --dir hei_rep
@REM FOR %%i IN (.2,.3,.4) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%w IN (1000) DO (
@REM             FOR %%v IN (.3) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 10 10 10 --topology geo  --dir hei_rep --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM             )
@REM         )
@REM     )
@REM )


@REM py -3.8 main.py --just_input True --length 100 --channels 700 --replicas 9 --patterns 3 --input_name "Heidelberg" --dir hei_repX
@REM FOR %%i IN (.12,.14,.16,.18,.2,.3,.4,.5) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%w IN (135) DO (
@REM             FOR %%v IN (.3) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --topology geo  --dir hei_X --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --rndp 111 --topology geo  --dir hei_X --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True

@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --topology smw  --dir hei_X --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --rndp 111 --topology smw  --dir hei_X --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True

@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp .3 --dir hei_X --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp .3 --beta 111 --dir hei_X --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM             )
@REM         )
@REM     )
@REM )

@REM ,.18,.19,.2,.25,.5)
@REM py -3.8 main.py --just_input True --length 100 --channels 700 --replicas 9 --patterns 3 --input_name "Heidelberg" --dir hei_repX
@REM FOR %%i IN (.17) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%y IN (0, 3) DO (
@REM             FOR %%x IN (0.0, 1.5) DO (
@REM                 FOR %%w IN (135) DO (
@REM                     FOR %%v IN (.1,.2,.3) DO (
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir hei_phei --length 700 --input_name "Heidelberg" --input_sparsity %%i
@REM                         FOR %%u IN (0.0, .33, .66) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir hei_phei --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                         )
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 15 3 3 --topology geo  --dir hei_phei --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 45 3 1 --topology geo  --dir hei_phei --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 9 5 3 --topology geo  --dir hei_phei --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )

@REM FINISH THIS SWEEP
@REM ,.18,.19,.2,.25,.5)
@REM py -3.8 main.py --just_input True --length 100 --channels 700 --replicas 9 --patterns 3 --input_name "Heidelberg" --dir hei_repX
@REM FOR %%i IN (.19,.2,.25,.5) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
        @REM FOR %%y IN (0) DO (
        @REM     FOR %%x IN (1.5) DO (
@REM                 FOR %%w IN (135) DO (
@REM                     FOR %%v IN (.1,.2,.3) DO (
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir hei_phei_rest --length 700 --input_name "Heidelberg" --input_sparsity %%i
@REM                         FOR %%u IN (0.0, .33, .66) DO (
@REM                             py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir hei_phei_rest --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                         )
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 15 3 3 --topology geo  --dir hei_phei_rest --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 45 3 1 --topology geo  --dir hei_phei_rest --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                         py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 9 5 3 --topology geo  --dir hei_phei_rest --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )


@REM @REM py -3.8 main.py --just_input True  --channels 850 --replicas 9 --patterns 9 --input_name "Heidelberg" --dir hei_large
@REM FOR %%i IN (.5) DO (
@REM     FOR %%z IN (Maass,STSP,STDP,LSTP) DO (
@REM         FOR %%w IN (500) DO (
@REM             FOR %%v IN (.3) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v --dir hei_large2 --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 9 --patterns 3
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 10 10 5 --topology geo  --dir hei_large2 --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 9 --patterns 3
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --topology smw --dir hei_large2 --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 9 --patterns 3
@REM             )
@REM         )
@REM     )
@REM )

@REM FOR %%i IN (.2,.3,.4,.5) DO (
@REM     FOR %%z IN (Maass,STSP,STDP,LSTP) DO (
@REM         FOR %%y IN (0,3) DO (
@REM             FOR %%x IN (0,1.5) DO (
@REM                 FOR %%w IN (135) DO (
@REM                     FOR %%v IN (.2,.3,.4,.5) DO (
@REM                         py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v --dir hei_large2 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3
@REM                         py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --topology geo  --dir hei_large2 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3
@REM                         py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --topology smw --dir hei_large2 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay %%x --refractory %%y --replicas 3 --patterns 3
@REM                     )
@REM                 )
@REM             )
@REM         )
@REM     )
@REM )





@REM hei_large3
@REM FOR %%i IN (.2,.3,.4,.5) DO (
@REM     FOR %%z IN (Maass,STSP,STDP,LSTP) DO (
@REM         FOR %%w IN (1000) DO (
@REM             FOR %%v IN (.2,.3,.4,.5) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp %%v --dir hei_large3 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 3 --patterns 3
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 10 10 10 --topology geo  --dir hei_large3 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 3 --patterns 3
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25  --topology smw --dir hei_large3 --length 700 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 3 --patterns 3
@REM             )
@REM         )
@REM     )
@REM )





@REM FOR %%i IN (.1,.2,.3,.4,.5) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%w IN (135) DO (
@REM             FOR %%v IN (.1,.2,.3,.4,.5) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --topology geo  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 9 --patterns 9
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 15 3 3 --rndp 111 --topology geo  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True --replicas 9 --patterns 9

@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --topology smw --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 9 --patterns 9
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.25 --rndp 111 --topology smw  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True --replicas 9 --patterns 9

@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp .3 --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0 --replicas 9 --patterns 9
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp .3 --beta 111 --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True --replicas 9 --patterns 9
@REM             )
@REM         )
@REM     )
@REM )
@REM FOR %%i IN (.2,.3,.4,.5) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%w IN (500) DO (
@REM             FOR %%v IN (.1,.2,.3,.4,.5) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 10 10 5 --topology geo  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 10 10 5 --rndp 111 --topology geo  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True 

@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.33 --topology smw  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --beta 0.33 --rndp 111 --topology smw  --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True

@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp .3 --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --delay 1.5 --refractory 0
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --topology rnd --rndp .3 --beta 111 --dir hei_large --length 800 --input_name "Heidelberg" --res_sparsity %%v --input_sparsity %%i --x_atory True
@REM             )
@REM         )
@REM     )
@REM )


@REM py -3.8 main.py --just_input True --length 100 --channels 40 --replicas 3 --patterns 2 --input_name "Poisson" --dir instant_poisson
@REM FOR %%i IN (.35) DO (
@REM     FOR %%z IN (Maass) DO (
@REM         FOR %%w IN (64) DO (
@REM             FOR %%v IN (.3) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 4 4 4 --topology geo  --dir instant_poisson --length 100 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i --replicas 3 --patterns 2

@REM             )
@REM         )
@REM     )
@REM )


@REM x_atory
@REM py -3.8 main.py --just_input True --length 500 --channels 40 --replicas 3 --patterns 3 --input_name "Poisson" --dir x_atory
@REM FOR %%i IN (.1,.2,.3) DO (
@REM     FOR %%z IN (Maass,STDP,STSP,LSTP) DO (
@REM         FOR %%w IN (64) DO (
@REM             FOR %%v IN (.1,.2,.3) DO (
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir x_atory --length 100 --input_name "Poisson" --input_sparsity %%i
@REM                 FOR %%u IN (0,0.5) DO (
@REM                     py -3.8 ./main.py --learning %%z  --neurons %%w --beta %%u --topology smw   --dir x_atory --length 100 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
@REM                 )
@REM                 py -3.8 ./main.py --learning %%z  --neurons %%w --dims 4 4 4 --topology geo  --dir x_atory --length 100 --input_name "Poisson" --res_sparsity %%v --input_sparsity %%i
@REM             )
@REM         )
@REM     )
@REM )


PAUSE
