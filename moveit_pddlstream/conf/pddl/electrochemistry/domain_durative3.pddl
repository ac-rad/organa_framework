(define (domain electrochemistry)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality :negative-preconditions); :durative-actions) ; :typing :conditional-effects

    ; (:types
    ;     electrode pre_target_motion_trajectory target_motion_trajectory post_target_motion_trajectory object_pose beaker liquid - object
    ;     solution - liquid

    ; )
    (:constants
        measurement_pose polishing_pose washing_pose ;- object_pose
        measurement_station ph_station ;- beaker
        pump_agent robot_agent measurement_agent ph_agent ; agents
        ; polishing_action washing_action moving_action transferring_action ; actions
        ; delta_time
        water buffer nacl quinone ; solution
        target_solution ; solution

        )

    ; (:functions
    ;     (total-cost)
    ;     (delta_time)
    ;     (polishing_time)
    ;     (washing_time)
    ;     (moving_time)
    ;     (transferring_time)
    ;     (current_time)
    ; )

    (:predicates
        ; dummy types
        (solution ?solution)
        (motion_trajectory ?motion_trajectory)
        (object_pose ?object_pose)
        (robot_state ?curr_state)
        (beaker ?beaker)
        ; (electrode ?electrode)
        ; (agent ?agent)
        ; (action ?action)
        (time ?time)
        ; (robot_time ?time)
        ; (pumpt_time ?time)
        ; (cost ?cost)

        ; electrochemistry predicates
        (polished)
        (washed)
        (at_pose ?object_pose)
        ; (measured ?electrode)
        ; (beaker_emptied ?beaker)
        (beaker_contains ?beaker ?solution)
        ; (solution_has ?target_solution ?primary_solution - liquid)
        ; (amount_liquid ?liquid)
        ; (beaker_has ?beaker ?liquid)
        (beaker_cleaned ?beaker)
        ; (characterized ?solution1 ?solution2)
        (characterized)
        (ph_measured)
        ; (solution_mixed ?solution1 ?solution2 ?solution3)
        (solution_mixed) ; if the target solution is mixed
        (is_busy ?agent)
        (at_state ?curr_state)
        ; (is_running ?action ?solution ?beaker)

        ; for streams
        ; (moveable ?electrode ?curr_state ?post_target_pose ?motion_trajectory ?target_state)
        (moveable ?curr_state ?target_pose ?post_target_state ?motion_trajectory)
        ; (time_updated ?old-time ?delta-time ?time)

        (at_time ?time)
        ; (robot_at_time ?time)
        ; (pump_at_time ?time)
        ; (at_time_cost ?cost)
        ; (cost_transfer_computed ?time ?old_sum_cost ?new_time ?new_sum_cost)
        ; (cost_move_computed ?time ?old_sum_cost ?new_time ?new_sum_cost)
        ; (cost_transfer_computed_start ?time ?new_time)
        (cost_transfer_computed ?time ?new_time)
        (cost_move_computed ?time ?new_time)
        (cost_polish_computed ?time ?new_time)
        (cost_wash_computed ?time ?new_time)
        (cost_measure_redux_potential_computed ?time ?new_time)
        (cost_measure_ph_computed ?time ?new_time)
        (cost_mix_solution_computed ?time ?new_time)
        (cost_empty_clean_measurement_station_computed ?time ?new_time)
        (cost_empty_clean_ph_station_computed ?time ?new_time)
        (cost_add_water_ph_station_computed ?time ?new_time)
        ; acitons running
        (running_move);?initial_state ?initial_pose ?target_pose ?post_target_state)
        (running_polish)
        (running_wash_electrode)
        (running_measure_redux_potential)
        (running_measure_ph)
        (running_transfer_liquid ?solution ?beaker)
        (running_transfer_quinone ?beaker)
        (running_transfer_target_solution ?beaker)
        (running_mix_solution)
        (running_empty_clean_measurement_station)
        (running_empty_clean_ph_station)
        (running_add_water_ph_station)

    )

    (:action move-start
        :parameters ( ?time ?initial_state ?initial_pose ?target_pose ?post_target_state ?motion_trajectory) ; 
        :precondition (and
            ; (initialized)
            ; (electrode ?electrode)
            (robot_state ?initial_state)
            (robot_state ?post_target_state)
            (object_pose ?initial_pose)
            (object_pose ?target_pose)
            (motion_trajectory ?motion_trajectory)
            (at_pose ?initial_pose)
            (at_state ?initial_state)
            ; (not (at_pose ?electrode ?target_pose))
            ; (not (= ?initial_pose ?target_pose))
            (moveable ?initial_state ?target_pose ?post_target_state ?motion_trajectory)
            ; cost and time update
            (time ?time)
            ; (time ?new_time)
            (at_time ?time)
            ; (not (at_time ?new_time))
            ; (cost_move_computed_start ?time ?new_time)
            ; durative
            (not (is_busy robot_agent))
            (not (running_move))

            ; (not (is_running moving_action))
        )
        :effect (and
            (not (at_pose ?initial_pose))
            (at_pose ?target_pose)
            (not (at_state ?initial_state))
            (at_state ?post_target_state)
            (is_busy robot_agent)
            (running_move)
            ; (is_running moving_action)
            ; (increase (total-cost) (moving_time))
            ; (not (at_time ?time))
            ; (at_time ?new_time)
            (increase
                (total-cost)
                (get_move_cost_start ?time))
        )
    )

    (:action move-end
        :parameters ( ?time ?new_time); ?initial_state ?initial_pose ?target_pose ?post_target_state ) 
        :precondition (and
            ; (initialized)
            ; (robot_state ?initial_state)
            ; (robot_state ?post_target_state)
            ; (object_pose ?initial_pose)
            ; (object_pose ?target_pose)
            ; (motion_trajectory ?motion_trajectory)
            ; (at_pose ?initial_pose)
            ; (at_state ?initial_state)
            ; (not (at_pose ?electrode ?target_pose))
            ; (not (= ?initial_pose ?target_pose))
            ; (moveable ?initial_state ?target_pose ?post_target_state ?motion_trajectory)
            ; cost and time update
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            ; (not (at_time ?new_time))
            (cost_move_computed ?time ?new_time)
            ; durative
            (is_busy robot_agent)
            (running_move)
            ; (is_running moving_action)
        )
        :effect (and
            ; (not (at_pose ?initial_pose))
            ; (at_pose ?target_pose)
            ; (not (at_state ?initial_state))
            ; (at_state ?post_target_state)
            (not (is_busy robot_agent))
            (not (running_move))
            ; (not (is_running moving_action))
            ; (increase (total-cost) (moving_time))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_move_cost_end))
        )
    )

    ; (:action move
    ;     :parameters ( ?time ?new_time ?initial_state ?initial_pose ?target_pose ?post_target_state ?motion_trajectory) ; 
    ;     :precondition (and
    ;         ; (initialized)
    ;         ; (electrode ?electrode)
    ;         (robot_state ?initial_state)
    ;         (robot_state ?post_target_state)
    ;         (object_pose ?initial_pose)
    ;         (object_pose ?target_pose)
    ;         (motion_trajectory ?motion_trajectory)
    ;         (at_pose ?initial_pose)
    ;         (at_state ?initial_state)
    ;         ; (not (at_pose ?electrode ?target_pose))
    ;         ; (not (= ?initial_pose ?target_pose))
    ;         (moveable ?initial_state ?target_pose ?post_target_state ?motion_trajectory)
    ;         ; (is_busy robot_agent)
    ;         ; (is_running moving_action)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_move_computed ?time ?new_time)
    ;     )
    ;     :effect (and
    ;         (not (at_pose ?initial_pose))
    ;         (at_pose ?target_pose)
    ;         (not (at_state ?initial_state))
    ;         (at_state ?post_target_state)
    ;         ; (not (is_busy robot_agent))
    ;         ; (not (is_running moving_action))
    ;         ; (increase (total-cost) (moving_time))
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         ; (increase
    ;         ;     (total-cost)
    ;         ;     (get_move_cost ?time ?new_time))
    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------

    (:action polish-start
        :parameters ( ?time)
        :precondition (and
            ; (electrode ?electrode)
            (not (polished))
            (at_pose polishing_pose)
            (washed)
            (time ?time)
            (at_time ?time)
            ; agents
            (not (is_busy robot_agent))
            (not (running_polish))
        )
        :effect (and
            ; (polished ?electrode)
            ; (not (washed ?electrode))
            (is_busy robot_agent)
            (running_polish)
            ; (not (at_time ?time))
            ; (at_time ?new_time)
            (increase
                (total-cost)
                (get_polish_cost_start ?time))
        )
    )

    (:action polish-end
        :parameters ( ?time ?new_time)
        :precondition (and
            ; (electrode ?electrode)
            (not (polished))
            (at_pose polishing_pose)
            (washed)
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_polish_computed ?time ?new_time)
            ; agents
            (is_busy robot_agent)
            (running_polish)
        )
        :effect (and
            (polished)
            (not (washed))
            (not (is_busy robot_agent))
            (not (running_polish))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_polish_cost_end))
        )
    )

    ; (:action polish
    ;     :parameters ( ?time ?new_time ?electrode)
    ;     :precondition (and
    ;         (electrode ?electrode)
    ;         (not (polished ?electrode))
    ;         (at_pose ?electrode polishing_pose)
    ;         (washed ?electrode)
    ;         ; (is_busy robot_agent)
    ;         ; (is_running polishing_action)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_polish_computed ?time ?new_time)
    ;         ; agents
    ;         (not (is_busy robot_agent))
    ;     )
    ;     :effect (and
    ;         (polished ?electrode)
    ;         (not (washed ?electrode))
    ;         ; (not (is_busy robot_agent))
    ;         ; (not (is_running polishing_action))
    ;         ; (increase (total-cost) (polishing_time))
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_polish_cost ?time ?new_time))

    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------
    (:action wash_electrode-start
        :parameters ( ?time)
        :precondition (and
            ; (electrode ?electrode)
            (at_pose washing_pose)
            (not (washed))
            (time ?time)
            (at_time ?time)
            ; agent
            (not (is_busy robot_agent))
            (not (running_wash_electrode))

        )
        :effect (and
            (is_busy robot_agent)
            (running_wash_electrode)
            (increase
                (total-cost)
                (get_wash_cost_start ?time))
        )
    )

    (:action wash_electrode-end
        :parameters ( ?time ?new_time)
        :precondition (and
            ; (electrode ?electrode)
            (at_pose washing_pose)
            (not (washed))
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_wash_computed ?time ?new_time)
            ; agent
            (is_busy robot_agent)
            (running_wash_electrode)
        )
        :effect (and
            (washed)
            (not (is_busy robot_agent))
            (not (running_wash_electrode))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_wash_cost_end))
        )
    )

    ; (:action wash_electrode
    ;     :parameters ( ?time ?new_time ?electrode)
    ;     :precondition (and
    ;         (electrode ?electrode)
    ;         ; (polished ?electrode)
    ;         (at_pose ?electrode washing_pose)
    ;         (not (washed ?electrode))
    ;         ; (is_busy robot_agent)
    ;         ; (is_running washing_action)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_wash_computed ?time ?new_time)
    ;         ; agent
    ;         (not (is_busy robot_agent))

    ;     )
    ;     :effect (and
    ;         (washed ?electrode)
    ;         ; (not (is_busy robot_agent))
    ;         ; (not (is_running washing_action))
    ;         ; (increase (total-cost) (washing_time))
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_wash_cost ?time ?new_time))
    ;     )
    ; )
    ;------------------------------------
    ;------------------------------------

    (:action measure_redux_potential-start
        :parameters ( ?time)
        :precondition (and
            ; (electrode ?electrode)
            (beaker_contains measurement_station target_solution)
            (polished)
            (washed)
            (at_pose measurement_pose)
            (solution_mixed)
            (not (characterized))
            (time ?time)
            (at_time ?time)
            ; agent
            (not (is_busy robot_agent))
            (not (is_busy measurement_agent))
            (not (running_measure_redux_potential))
        )
        :effect (and
            ; (not (polished ?electrode))
            ; (not (washed ?electrode))
            ; (characterized)
            (is_busy robot_agent)
            (is_busy measurement_agent)
            (running_measure_redux_potential)
            (increase
                (total-cost)
                (get_measure_redux_potential_cost_start ?time))
        )
    )

    (:action measure_redux_potential-end
        :parameters ( ?time ?new_time)
        :precondition (and
            ; (electrode ?electrode)
            (beaker_contains measurement_station target_solution)
            (polished)
            (washed)
            (at_pose measurement_pose)
            (solution_mixed)
            (not (characterized))
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_measure_redux_potential_computed ?time ?new_time)
            ; agent
            (is_busy robot_agent)
            (is_busy measurement_agent)
            (running_measure_redux_potential)
        )
        :effect (and
            (not (polished))
            (not (washed))
            (characterized)
            ; agent
            (not (is_busy robot_agent))
            (not (is_busy measurement_agent))
            (not (running_measure_redux_potential))
            ; time
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_measure_redux_potential_cost_end))
        )
    )

    ; (:action measure_redux_potential
    ;     :parameters ( ?time ?new_time ?electrode)
    ;     :precondition (and
    ;         (electrode ?electrode)
    ;         ; (solution ?solution1)
    ;         ; (solution ?solution2)
    ;         ; (solution ?solution3)
    ;         (beaker_contains measurement_station target_solution)
    ;         (polished ?electrode)
    ;         (washed ?electrode)
    ;         ; (not (measured ?electrode))
    ;         (at_pose ?electrode measurement_pose)
    ;         ; (solution_mixed ?solution1 ?solution2 ?solution3)
    ;         (solution_mixed)
    ;         ; (not (characterized ?solution1 ?solution2))
    ;         ; (not (characterized ?solution2 ?solution1))
    ;         (not (characterized))
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_measure_redux_potential_computed ?time ?new_time)
    ;         ; agent
    ;         (not (is_busy robot_agent))
    ;     )
    ;     :effect (and
    ;         (not (polished ?electrode))
    ;         (not (washed ?electrode))
    ;         ; (measured ?electrode)
    ;         ; (characterized ?solution1 ?solution2)
    ;         ; (characterized ?solution2 ?solution1)
    ;         (characterized)
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_measure_redux_potential_cost ?time ?new_time))

    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------ 
    (:action measure_ph-start
        :parameters ( ?time)
        :precondition (and
            (beaker_contains ph_station target_solution)
            (not (beaker_contains ph_station water))
            (not (beaker_contains ph_station quinone))
            (not (beaker_contains ph_station buffer))
            (not (beaker_contains ph_station nacl))
            (characterized)
            (solution_mixed)
            (time ?time)
            (at_time ?time)
            (not (is_busy ph_agent))
            (not (running_measure_ph))
        )
        :effect (and
            (is_busy ph_agent)
            (running_measure_ph)
            (increase
                (total-cost)
                (get_measure_ph_cost_start ?time))
        )
    )
    (:action measure_ph-end
        :parameters ( ?time ?new_time)
        :precondition (and
            (beaker_contains ph_station target_solution)
            (not (beaker_contains ph_station water))
            (not (beaker_contains ph_station quinone))
            (not (beaker_contains ph_station buffer))
            (not (beaker_contains ph_station nacl))
            (characterized)
            (solution_mixed)
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_measure_ph_computed ?time ?new_time)
            (is_busy ph_agent)
            (running_measure_ph)
        )
        :effect (and
            (ph_measured)
            (not (at_time ?time))
            (at_time ?new_time)
            (not (is_busy ph_agent))
            (not (running_measure_ph))
            (increase
                (total-cost)
                (get_measure_ph_cost_end))
        )
    )

    ; (:action measure_ph
    ;     :parameters ( ?time ?new_time)
    ;     :precondition (and
    ;         (beaker_contains ph_station target_solution)
    ;         (not (beaker_contains ph_station water))
    ;         (not (beaker_contains ph_station quinone))
    ;         (not (beaker_contains ph_station buffer))
    ;         (not (beaker_contains ph_station nacl))
    ;         (characterized)
    ;         (solution_mixed)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_measure_ph_computed ?time ?new_time)

    ;     )
    ;     :effect (and
    ;         (ph_measured)
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_measure_ph_cost ?time ?new_time))
    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------

    (:action transfer_liquid-start
        :parameters (?time ?solution ?beaker)
        :precondition (and
            ; (initialized)
            (solution ?solution)
            (beaker ?beaker)
            (not (= ?solution quinone))
            (not (= ?solution target_solution))
            ; (is_running transferring_action)
            ; (is_busy pump_agent)
            (time ?time)
            ; (time ?new_time)
            ; (time ?old_sum_cost)
            ; (time ?new_sum_cost)
            ; (time ?cost)
            (at_time ?time)
            ; (not (at_time ?new_time))
            ; (at_time_cost ?old_sum_cost)
            ; (cost_transfer_computed_start ?time ?new_time)
            ; agent
            (not (is_busy pump_agent))
            (not (running_transfer_liquid ?solution ?beaker))
        )
        :effect (and
            ; (beaker_contains ?beaker ?solution)
            ; (not (beaker_cleaned ?beaker))
            (is_busy pump_agent)
            (running_transfer_liquid ?solution ?beaker)
            ; (increase (total-cost) (transferring_time))
            ; (not (at_time ?time))
            ; (at_time ?new_time)
            (increase
                (total-cost)
                (get_transfer_cost_start ?time))
        )
    )
    (:action transfer_liquid-end
        :parameters (?time ?new_time ?solution ?beaker)
        :precondition (and
            ; (initialized)
            (solution ?solution)
            (beaker ?beaker)
            (not (= ?solution quinone))
            (not (= ?solution target_solution))
            ; (is_running transferring_action)
            ; (is_busy pump_agent)
            (time ?time)
            (time ?new_time)
            ; (time ?old_sum_cost)
            ; (time ?new_sum_cost)
            ; (time ?cost)
            (at_time ?time)
            ; (not (at_time ?new_time))
            ; (at_time_cost ?old_sum_cost)
            (cost_transfer_computed ?time ?new_time)
            ; agent
            (is_busy pump_agent)
            (running_transfer_liquid ?solution ?beaker)
        )
        :effect (and
            (beaker_contains ?beaker ?solution)
            (not (beaker_cleaned ?beaker))
            (not (is_busy pump_agent))
            (not (running_transfer_liquid ?solution ?beaker))
            ; (increase (total-cost) (transferring_time))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_transfer_cost_end))
        )
    )

    ; (:action transfer_liquid
    ;     :parameters (?time ?new_time ?solution ?beaker)
    ;     :precondition (and
    ;         ; (initialized)
    ;         (solution ?solution)
    ;         (beaker ?beaker)
    ;         (not (= ?solution quinone))
    ;         (not (= ?solution target_solution))
    ;         ; (is_running transferring_action)
    ;         ; (is_busy pump_agent)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         ; (time ?old_sum_cost)
    ;         ; (time ?new_sum_cost)
    ;         ; (time ?cost)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         ; (at_time_cost ?old_sum_cost)
    ;         (cost_transfer_computed ?time ?new_time)
    ;     )
    ;     :effect (and
    ;         (beaker_contains ?beaker ?solution)
    ;         (not (beaker_cleaned ?beaker))
    ;         ; (not (is_busy pump_agent))
    ;         ; (not (is_running transferring_action))
    ;         ; (increase (total-cost) (transferring_time))
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_transfer_cost ?time ?new_time))
    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------ 

    (:action transfer_quinone-start
        :parameters ( ?time ?beaker);?time ?new_time ?cost
        :precondition (and
            (beaker ?beaker)
            (beaker_contains ?beaker water)
            (beaker_contains ?beaker nacl)
            (beaker_contains ?beaker buffer)
            (time ?time)
            (at_time ?time)
            ; agent
            (not (is_busy pump_agent))
            (not (running_transfer_quinone ?beaker))
        )
        :effect (and
            (is_busy pump_agent)
            (running_transfer_quinone ?beaker)
            (increase
                (total-cost)
                (get_transfer_cost_start ?time))
        )
    )
    (:action transfer_quinone-end
        :parameters ( ?time ?new_time ?beaker);?time ?new_time ?cost
        :precondition (and
            (beaker ?beaker)
            (beaker_contains ?beaker water)
            (beaker_contains ?beaker nacl)
            (beaker_contains ?beaker buffer)
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_transfer_computed ?time ?new_time)
            ; agent
            (is_busy pump_agent)
            (running_transfer_quinone ?beaker)

        )
        :effect (and
            (beaker_contains ?beaker quinone)
            (not (beaker_cleaned ?beaker))
            (not (is_busy pump_agent))
            (not (running_transfer_quinone ?beaker))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_transfer_cost_end))
        )
    )

    ; (:action transfer_quinone
    ;     :parameters ( ?time ?new_time ?beaker);?time ?new_time ?cost
    ;     :precondition (and
    ;         (beaker ?beaker)
    ;         (beaker_contains ?beaker water)
    ;         (beaker_contains ?beaker nacl)
    ;         (beaker_contains ?beaker buffer)
    ;         ; (is_running transferring_action)
    ;         ; (is_busy pump_agent)
    ;         ; (time ?time)
    ;         ; (time ?new_time)
    ;         ; (at_time ?time)
    ;         ; (cost_transfer_computed ?time ?new_time ?cost)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_transfer_computed ?time ?new_time)
    ;         (not (is_busy pump_agent))

    ;     )
    ;     :effect (and
    ;         (beaker_contains ?beaker quinone)
    ;         (not (beaker_cleaned ?beaker))
    ;         ; (not (is_busy pump_agent))
    ;         ; (not (is_running transferring_action))
    ;         ; (increase (total-cost) (transferring_time))
    ;         ; (not (at_time ?time))
    ;         ; (at_time ?new_time)
    ;         ; (increase (total-cost) (compute_cost ?cost))
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_transfer_cost ?time ?new_time))
    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------
    (:action transfer_target_solution-start
        :parameters ( ?time ?beaker);?time ?new_time ?cost
        :precondition (and
            (beaker ?beaker)
            (solution_mixed)
            (beaker_cleaned ?beaker)
            (not (beaker_cleaned measurement_station))
            (forall(?solution)
                (and
                    (solution ?solution)
                    (not (beaker_contains ?beaker ?solution))
                )
            )
            (time ?time)
            (at_time ?time)
            ; agent
            (not (is_busy pump_agent))
            (not (running_transfer_target_solution ?beaker))

        )
        :effect (and
            (is_busy pump_agent)
            (running_transfer_target_solution ?beaker)
            (increase
                (total-cost)
                (get_transfer_cost_start ?time))
        )
    )
    (:action transfer_target_solution-end
        :parameters ( ?time ?new_time ?beaker);?time ?new_time ?cost
        :precondition (and
            (beaker ?beaker)
            (solution_mixed)
            (beaker_cleaned ?beaker)
            (not (beaker_cleaned measurement_station))
            (forall(?solution)
                (and
                    (solution ?solution)
                    (not (beaker_contains ?beaker ?solution))
                )
            )
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_transfer_computed ?time ?new_time)
            ; agent
            (is_busy pump_agent)
            (running_transfer_target_solution ?beaker)
        )
        :effect (and
            (beaker_contains ?beaker target_solution)
            (not (beaker_cleaned ?beaker))
            (not (is_busy pump_agent))
            (not (running_transfer_target_solution ?beaker))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_transfer_cost_end))
        )
    )

    ; ; (:action transfer_target_solution
    ; ;     :parameters ( ?time ?new_time ?beaker);?time ?new_time ?cost
    ; ;     :precondition (and
    ; ;         (beaker ?beaker)
    ; ;         (solution_mixed)
    ; ;         (beaker_cleaned ?beaker)
    ; ;         (not (beaker_cleaned measurement_station))
    ; ;         (forall(?solution)
    ; ;             (and
    ; ;                 (solution ?solution)
    ; ;                 (not (beaker_contains ?beaker ?solution))
    ; ;             )
    ; ;         )
    ; ;         ; (is_running transferring_action)
    ; ;         ; (is_busy pump_agent)
    ; ;         ; (time ?time)
    ; ;         ; (time ?new_time)
    ; ;         ; (at_time ?time)
    ; ;         ; (cost_transfer_computed ?time ?new_time ?cost)
    ; ;         (time ?time)
    ; ;         (time ?new_time)
    ; ;         (at_time ?time)
    ; ;         (not (at_time ?new_time))
    ; ;         (cost_transfer_computed ?time ?new_time)
    ; ;         (not (is_busy pump_agent))
    ; ;     )
    ; ;     :effect (and
    ; ;         (beaker_contains ?beaker target_solution)
    ; ;         (not (beaker_cleaned ?beaker))
    ; ;         ; (not (is_busy pump_agent))
    ; ;         ; (not (is_running transferring_action))
    ; ;         ; (increase (total-cost) (transferring_time))
    ; ;         ; (not (at_time ?time))
    ; ;         ; (at_time ?new_time)
    ; ;         ; (increase (total-cost) (compute_cost ?cost))
    ; ;         (not (at_time ?time))
    ; ;         (at_time ?new_time)
    ; ;         (increase
    ; ;             (total-cost)
    ; ;             (get_transfer_cost ?time ?new_time))
    ; ;     )
    ; ; )

    ;------------------------------------
    ;------------------------------------

    ; mix the solutions exists in a beaker
    (:action mix_solution-start
        :parameters ( ?time);?time ?new_time ?cost
        :precondition (and
            (beaker_contains measurement_station water)
            (beaker_contains measurement_station nacl)
            (beaker_contains measurement_station buffer)
            (beaker_contains measurement_station quinone)
            (time ?time)
            (at_time ?time)
            (not (is_busy pump_agent))
            (not(running_mix_solution))
        )
        :effect (and
            (is_busy pump_agent)
            (running_mix_solution)
            (increase
                (total-cost)
                (get_mix_solution_cost_start ?time))
        )
    )
    (:action mix_solution-end
        :parameters ( ?time ?new_time);?time ?new_time ?cost
        :precondition (and
            (beaker_contains measurement_station water)
            (beaker_contains measurement_station nacl)
            (beaker_contains measurement_station buffer)
            (beaker_contains measurement_station quinone)
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_mix_solution_computed ?time ?new_time)
            (is_busy pump_agent)
            (running_mix_solution)
        )
        :effect (and
            (solution_mixed)
            (beaker_contains measurement_station target_solution)
            (not (at_time ?time))
            (at_time ?new_time)
            (not (is_busy pump_agent))
            (not (running_mix_solution))
            (increase
                (total-cost)
                (get_mix_solution_cost_end))
        )
    )

    ; (:action mix_solution
    ;     :parameters ( ?time ?new_time);?time ?new_time ?cost
    ;     :precondition (and
    ;         (beaker_contains measurement_station water)
    ;         (beaker_contains measurement_station nacl)
    ;         (beaker_contains measurement_station buffer)
    ;         (beaker_contains measurement_station quinone)
    ;         ; (time ?time)
    ;         ; (time ?new_time)
    ;         ; (at_time ?time)
    ;         ; (cost_transfer_computed ?time ?new_time ?cost)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_mix_solution_computed ?time ?new_time)
    ;         (not (is_busy pump_agent))
    ;     )
    ;     :effect (and
    ;         ; (solution_mixed ?solution1 ?solution2 ?solution3)
    ;         ; (solution_mixed ?solution2 ?solution1 ?solution3)
    ;         (solution_mixed)
    ;         (beaker_contains measurement_station target_solution)
    ;         ; (not (at_time ?time))
    ;         ; (at_time ?new_time)
    ;         ; (increase (total-cost) (compute_cost ?cost))
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_mix_solution_cost ?time ?new_time))
    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------
    ; empty and clean the beaker by
    (:action empty_clean_measurement_station-start
        :parameters ( ?time)
        :precondition (and
            (not (beaker_cleaned measurement_station))
            (time ?time)
            (at_time ?time)
            (not (is_busy pump_agent))
            (not (running_empty_clean_measurement_station))
        )
        :effect (and
            (is_busy pump_agent)
            (running_empty_clean_measurement_station)
            (increase
                (total-cost)
                (get_empty_clean_measurement_station_cost_start ?time))
        )
    )
    (:action empty_clean_measurement_station-end
        :parameters ( ?time ?new_time)
        :precondition (and
            (not (beaker_cleaned measurement_station))
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_empty_clean_measurement_station_computed ?time ?new_time)
            (is_busy pump_agent)
            (running_empty_clean_measurement_station)
        )
        :effect (and
            (beaker_cleaned measurement_station)
            (forall(?solution)
                (and
                    (solution ?solution)
                    (not (beaker_contains measurement_station ?solution))
                )
            )
            (not (is_busy pump_agent))
            (not (running_empty_clean_measurement_station))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_empty_clean_measurement_station_cost_end))

        )
    )

    ; (:action empty_clean_measurement_station
    ;     :parameters ( ?time ?new_time)
    ;     :precondition (and
    ;         ; (not (beaker_emptied ?beaker))
    ;         ; (beaker ?beaker)
    ;         ; (solution ?solution1)
    ;         ; (solution ?solution2)
    ;         (not (beaker_cleaned measurement_station))
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_empty_clean_measurement_station_computed ?time ?new_time)
    ;         (not (is_busy pump_agent))
    ;     )
    ;     :effect (and
    ;         (beaker_cleaned measurement_station)
    ;         ; (beaker_emptied ?beaker)
    ;         (forall(?solution)
    ;             (and
    ;                 (solution ?solution)
    ;                 (not (beaker_contains measurement_station ?solution))
    ;             )
    ;         )
    ;         ; (not (solution_mixed))

    ;         ; (forall
    ;         ;     (?solution1 ?solution2 ?solution3)
    ;         ;     ; (when (and
    ;         ;     ;     (beaker_contains ?beaker ?solution1)
    ;         ;     ;     (beaker_contains ?beaker ?solution2)
    ;         ;     ;     )
    ;         ;     (and
    ;         ;         (solution ?solution1)
    ;         ;         (solution ?solution2)
    ;         ;         (solution ?solution3)
    ;         ;         (not (solution_mixed ?solution1 ?solution2 ?solution3))
    ;         ;         (not (solution_mixed ?solution2 ?solution1 ?solution3))
    ;         ;     )
    ;         ;     ; )
    ;         ; )
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_empty_clean_measurement_station_cost ?time ?new_time))

    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------
    (:action empty_clean_ph_station-start
        :parameters ( ?time)
        :precondition (and
            (not (beaker_cleaned ph_station))
            (characterized)
            (time ?time)
            (at_time ?time)
            (not (is_busy pump_agent))
            (not (running_empty_clean_ph_station))
        )
        :effect (and
            (is_busy pump_agent)
            (running_empty_clean_ph_station)
            (increase
                (total-cost)
                (get_empty_clean_ph_station_cost_start ?time))
        )
    )
    (:action empty_clean_ph_station-end
        :parameters ( ?time ?new_time)
        :precondition (and
            (not (beaker_cleaned ph_station))
            (characterized)
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_empty_clean_ph_station_computed ?time ?new_time)
            (is_busy pump_agent)
            (running_empty_clean_ph_station)
        )
        :effect (and
            (beaker_cleaned ph_station)
            (forall(?solution)
                (and
                    (solution ?solution)
                    (not (beaker_contains ph_station ?solution))
                )
            )
            (not (is_busy pump_agent))
            (not (running_empty_clean_ph_station))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_empty_clean_ph_station_cost_end))
        )
    )

    ; (:action empty_clean_ph_station
    ;     :parameters ( ?time ?new_time)
    ;     :precondition (and
    ;         (not (beaker_cleaned ph_station))
    ;         (characterized)
    ;         ; (not (ph_measured))
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_empty_clean_ph_station_computed ?time ?new_time)
    ;         (not (is_busy pump_agent))
    ;     )
    ;     :effect (and
    ;         (beaker_cleaned ph_station)
    ;         (forall(?solution)
    ;             (and
    ;                 (solution ?solution)
    ;                 (not (beaker_contains ph_station ?solution))
    ;             )
    ;         )
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_empty_clean_ph_station_cost ?time ?new_time))
    ;     )
    ; )

    ;------------------------------------
    ;------------------------------------
    (:action add_water_ph_station-start
        :parameters ( ?time)
        :precondition (and
            (beaker_cleaned ph_station)
            (characterized)
            (ph_measured)
            (beaker_cleaned ph_station)
            (time ?time)
            (at_time ?time)
            (not (is_busy pump_agent))
            (not (running_add_water_ph_station))
        )
        :effect (and
            (is_busy pump_agent)
            (running_add_water_ph_station)
            (increase
                (total-cost)
                (get_add_water_ph_station_cost_start ?time))
        )
    )
    (:action add_water_ph_station-end
        :parameters ( ?time ?new_time)
        :precondition (and
            (beaker_cleaned ph_station)
            (characterized)
            (ph_measured)
            (beaker_cleaned ph_station)
            (time ?time)
            (time ?new_time)
            (at_time ?time)
            (not (at_time ?new_time))
            (cost_add_water_ph_station_computed ?time ?new_time)
            (is_busy pump_agent)
            (running_add_water_ph_station)
        )
        :effect (and
            (beaker_contains ph_station water)
            (not (is_busy pump_agent))
            (not (running_add_water_ph_station))
            (not (at_time ?time))
            (at_time ?new_time)
            (increase
                (total-cost)
                (get_add_water_ph_station_cost_end))
        )
    )

    ; (:action add_water_ph_station
    ;     :parameters ( ?time ?new_time)
    ;     :precondition (and
    ;         (beaker_cleaned ph_station)
    ;         (characterized)
    ;         (ph_measured)
    ;         (beaker_cleaned ph_station)
    ;         (time ?time)
    ;         (time ?new_time)
    ;         (at_time ?time)
    ;         (not (at_time ?new_time))
    ;         (cost_add_water_ph_station_computed ?time ?new_time)
    ;         (not (is_busy pump_agent))
    ;     )
    ;     :effect (and
    ;         (beaker_contains ph_station water)
    ;         (not (at_time ?time))
    ;         (at_time ?new_time)
    ;         (increase
    ;             (total-cost)
    ;             (get_add_water_ph_station_cost ?time ?new_time))
    ;     )
    ; )
    ;------------------------------------
    ;------------------------------------

)