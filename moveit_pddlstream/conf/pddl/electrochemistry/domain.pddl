(define (domain electrochemistry)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality :negative-preconditions) ; :typing :conditional-effects

    ; (:types
    ;     electrode pre_target_motion_trajectory target_motion_trajectory post_target_motion_trajectory object_pose beaker liquid - object
    ;     solution - liquid

    ; )
    (:constants
        measurement_pose polishing_pose washing_pose ;- object_pose
        measurement_station ph_station;- beaker
        pump_agent robot_agent ; agents
        polishing_action washing_action moving_action transferring_action ; actions
        ; delta_time

        )

    (:functions
        (total-cost)
        (delta_time)
        (polishing_time)
        (washing_time)
        (moving_time)
        (transferring_time)
    )

    (:predicates
        ; dummy types
        (solution ?solution)
        (motion_trajectory ?motion_trajectory)
        (object_pose ?object_pose)
        (robot_state ?curr_state)
        (beaker ?beaker)
        (electrode ?electrode)
        (agent ?agent)
        (action ?action)
        (time ?time)

        ; electrochemistry predicates
        (polished ?electrode)
        (washed ?electrode)
        (at_pose ?object ?object_pose)
        ; (measured_ph ?solution1 ?solution2 ?solution3 ?solution4)

        ; (beaker_emptied ?beaker)
        (beaker_contains ?beaker ?solution)
        ; (solution_has ?target_solution ?primary_solution - liquid)
        ; (amount_liquid ?liquid)
        ; (beaker_has ?beaker ?liquid)
        (beaker_cleaned ?beaker)
        (characterized ?solution1 ?solution2 ?solution3 ?solution4)
        (solution_mixed ?solution1 ?solution2 ?solution3 ?solution4)
        (is_busy ?agent)
        (at_state ?curr_state)
        (is_running ?action)

        ; for streams
        ; (moveable ?electrode ?curr_state ?post_target_pose ?motion_trajectory ?target_state)
        (moveable ?curr_state ?target_pose ?post_target_state ?motion_trajectory)
        ; (time_updated ?old-time ?delta-time ?time)
    )

    (:action move
        :parameters ( ?electrode ?initial_state ?initial_pose ?target_pose ?post_target_state ?motion_trajectory)
        :precondition (and
            (electrode ?electrode)
            (robot_state ?initial_state)
            (robot_state ?post_target_state)
            (object_pose ?initial_pose)
            (object_pose ?target_pose)
            (motion_trajectory ?motion_trajectory)
            (at_pose ?electrode ?initial_pose)
            (at_state ?initial_state)
            (moveable  ?initial_state ?target_pose ?post_target_state ?motion_trajectory)
        )
        :effect (and
            (not (at_pose ?electrode ?initial_pose))
            (at_pose ?electrode ?target_pose)
            (not (at_state ?initial_state))
            (at_state ?post_target_state)
        )
    )

    (:action polish
        :parameters ( ?electrode)
        :precondition (and
            (electrode ?electrode)
            (not (polished ?electrode))
            ; (washed ?electrode)
            (at_pose ?electrode polishing_pose)
        )
        :effect (and
            (polished ?electrode)
            ; (not (washed ?electrode))
        )
    )

    (:action wash_electrode
        :parameters (?electrode)
        :precondition (and
            (electrode ?electrode)
            ; (polished ?electrode)
            (at_pose ?electrode washing_pose)
            (not (washed ?electrode))
        )
        :effect (and
            (washed ?electrode)
        )
    )

    (:action measure_redux_potential
        :parameters ( ?electrode ?solution1 ?solution2 ?solution3 ?solution4)
        :precondition (and
            (electrode ?electrode)
            (solution ?solution1)
            (solution ?solution2)
            (solution ?solution3)
            (solution ?solution4)

            (polished ?electrode)
            (washed ?electrode)
            (at_pose ?electrode measurement_pose)
            (solution_mixed ?solution1 ?solution2 ?solution3 ?solution4)
            (not (characterized ?solution1 ?solution2 ?solution3 ?solution4 ))
            ; (not (characterized ?solution2 ?solution1))
        )
        :effect (and
            (not (polished ?electrode))
            (not (washed ?electrode))
            (characterized ?solution1 ?solution2  ?solution3 ?solution4)
            ; (characterized ?solution2 ?solution1)
        )
    )


    ; (:action measure_ph
    ;     :parameters ( ?solution1 ?solution2 ?solution3 ?solution4)
    ;     :precondition (and
    ;         (solution ?solution1)
    ;         (solution ?solution2)
    ;         (solution ?solution3)
    ;         (solution ?solution4)

    ;         (solution_mixed ?solution1 ?solution2 ?solution3 ?solution4)
    ;         (not (characterized ?solution1 ?solution2 ?solution3 ?solution4 ))
    ;         (not (measured_ph ?solution1 ?solution2 ?solution3 ?solution4 ))
    ;         ; (not (characterized ?solution2 ?solution1))
    ;     )
    ;     :effect (and
    ;         (measured_ph ?solution1 ?solution2  ?solution3 ?solution4)
    ;     )
    ; )


    (:action transfer_liquid
        :parameters (?solution ?beaker)
        :precondition (and
            (solution ?solution)
            (beaker ?beaker)
        )
        :effect (and
            (beaker_contains ?beaker ?solution)
            (not (beaker_cleaned ?beaker))
            (increase (total-cost) (transferring_time))
        )
    )

    ; mix the solutions exists in a beaker
    (:action mix_solution
        :parameters (?solution1 ?solution2 ?solution3 ?solution4)
        :precondition (and
            (solution ?solution1)
            (solution ?solution2)
            (solution ?solution3)
            (solution ?solution4)

            (beaker_contains measurement_station ?solution1)
            (beaker_contains measurement_station ?solution2)
            (beaker_contains measurement_station ?solution3)
            (beaker_contains measurement_station ?solution4)
        )
        :effect (and
            (solution_mixed ?solution1 ?solution2 ?solution3 ?solution4 )
            ; (solution_mixed ?solution2 ?solution1)
        )
    )

    ; empty and clean the beaker by
    (:action empty_clean_beaker
        :parameters (?beaker)
        :precondition (and
            ; (not (beaker_emptied ?beaker))
            (beaker ?beaker)
            ; (solution ?solution1)
            ; (solution ?solution2)
            (not (beaker_cleaned ?beaker))
        )
        :effect (and
            (beaker_cleaned ?beaker)
            ; (beaker_emptied ?beaker)
            (forall(?solution)
                (and
                    (solution ?solution)
                    (not (beaker_contains ?beaker ?solution))
                )
            )

            (forall
                (?solution1 ?solution2)
                ; (when (and
                ;     (beaker_contains ?beaker ?solution1)
                ;     (beaker_contains ?beaker ?solution2)
                ;     )
                (and
                    (solution ?solution1)
                    (solution ?solution2)
                    (not (solution_mixed ?solution1 ?solution2 ?solution3 ?solution4))
                    ; (not (solution_mixed ?solution2 ?solution1))
                )
                ; )
            )
        )
    )
)