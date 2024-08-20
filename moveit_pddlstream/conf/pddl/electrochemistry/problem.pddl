(define (problem electrochemistry)
    (:domain electrochemistry)

    (:objects
        object - object
        water_container buffer_container quinone_container wasting_station ph_station - beaker
        electrode - electrode
        measurement_solution quinone buffer water - solution
        ;  - liquid
        ; polishing_pose - polishing_pose
        ; washing_pose - washing_pose
        ; measurement_pose - measurement_pose
        home_pose - object_pose
    )

    (:init
        (not (polished electrode))
        (not (washed electrode))
        (at_pose electrode home_pose)
        (beaker_contains quinone_container quinone)
        (beaker_contains buffer_container buffer)
        (beaker_contains water_container water)

        ; (beaker_contains measurement_station quinone)
        ; (beaker_contains measurement_station buffer)

        ; (not (beaker_cleaned measurement_station))
        ; (not (beaker_cleaned ph_station))

        ; (not (= home_pose polishing_pose))

    )

    ; define as many `said_hello_to` goals as you have neighbors
    (:goal
        (and
            ; greeted everyone else
            (at_pose electrode polishing_pose)
            ; (polished electrode)
            ; (washed electrode)
            ; (beaker_contains measurement_station quinone)
            ; (beaker_cleaned measurement_station)
            ; (beaker_cleaned ph_station)
            ; (solution_has measurement_solution quinone)
            ; (solution_has measurement_solution buffer)

            ; (characterized quinone buffer)
            ; (beaker_cleaned measurement_station)
        )
    )
    ; (:metric minimize (total-time))
)

; TODO: make the liquid transfering and polishing/washin actions in parallel. 