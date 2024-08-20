(define
    (stream electrochemistry)
    (:stream find_move
        :inputs
        ( ?initial_state ?target_pose); ?electrode ?initial_pose 
        :domain
        (and
            (object_pose ?target_pose)
            (robot_state ?initial_state)
        )
        :outputs
        (?post_target_state ?motion_trajectory) ; only continuous
        :certified
        (and
            (moveable ?initial_state ?target_pose ?post_target_state ?motion_trajectory) ; only logical
            (motion_trajectory ?motion_trajectory)
            (robot_state ?post_target_state)
        )
    )

    ; (:stream sampler
    ;     :outputs (?s)
    ;     :certified (sampler ?s)
    ; )

    ; (:stream cost_update
    ;     :inputs
    ;     ( ?time); ?electrode ?initial_pose 
    ;     :domain
    ;     (and
    ;         (time ?time)
    ;     )
    ;     :outputs
    ;     (?cost ?sampler) ; only continuous
    ;     :certified
    ;     (and
    ;         (cost_computed ?time ?cost ?sampler) ; only logical
    ;         (time ?cost)
    ;         (sampler ?sampler)
    ;     )
    ; )

    ;----------------------------
    (:stream cost_transfer_update
        :inputs
        (?time); ?electrode ?initial_pose 
        :domain
        (and
            (time ?time)
            ; (time ?old_sum_cost)
            ; (beaker ?beaker)
            ; (solution ?solution)
        )
        :outputs
        (?new_time) ; only continuous
        :certified
        (and
            (cost_transfer_computed ?time ?new_time) ; only logical
            (time ?new_time)
            ; (time ?cost)
            ; (time ?new_sum_cost)
        )
    )
    (:function
        (get_transfer_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_transfer_computed ?time ?new_time))
    )
    ;----------------------------

    (:stream cost_move_update
        :inputs
        (?time); ?electrode ?initial_pose  ?time ?old_sum_cost ?new_time ?cost ?new_sum_cost
        :domain
        (and
            (time ?time)
            ; (time ?delta_time)
            ; (time ?old_sum_cost)
            ; (beaker ?beaker)
            ; (solution ?solution)
        )
        :outputs
        (?new_time) ; only continuous
        :certified
        (and
            (cost_move_computed ?time ?new_time) ; only logical
            (time ?new_time)
            ; (time ?cost)
            ; (time ?new_sum_cost)
        )
    )
    (:function
        (get_move_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_move_computed ?time ?new_time))
    )
    ; ----------------------
    (:stream cost_polish_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_polish_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_polish_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_polish_computed ?time ?new_time))
    )
    ; ----------------------
    (:stream cost_wash_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_wash_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_wash_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_wash_computed ?time ?new_time))
    )
    ; ----------------------
    (:stream cost_measure_redux_potential_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_measure_redux_potential_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_measure_redux_potential_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_measure_redux_potential_computed ?time ?new_time))
    )
    ; ----------------------
    (:stream cost_measure_ph_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_measure_ph_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_measure_ph_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_measure_ph_computed ?time ?new_time))
    )
    ; ----------------------
    (:stream cost_mix_solution_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_mix_solution_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_mix_solution_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_mix_solution_computed ?time ?new_time))
    )

    ; ; ----------------------
    (:stream cost_empty_clean_measurement_station_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_empty_clean_measurement_station_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_empty_clean_measurement_station_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_empty_clean_measurement_station_computed ?time ?new_time))
    )
    ; ----------------------
    (:stream cost_empty_clean_ph_station_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_empty_clean_ph_station_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_empty_clean_ph_station_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_empty_clean_ph_station_computed ?time ?new_time))
    ) ; ----------------------
    (:stream cost_add_water_ph_station_update
        :inputs
        (?time)
        :domain
        (and
            (time ?time)
        )
        :outputs
        (?new_time)
        :certified
        (and
            (cost_add_water_ph_station_computed ?time ?new_time)
            (time ?new_time)
        )
    )
    (:function
        (get_add_water_ph_station_cost ?time ?new_time)
        (and (time ?time) (time ?new_time) (cost_add_water_ph_station_computed ?time ?new_time))
    )
    ; ----------------------

    ; ----------------------

    ; (:function (compute_cost ?time ?cost ?sampler)
    ;      (and (time ?time) (time ?cost) (sampler ?sampler))
    ; )

    ; (:function (get_transfer_cost ?solution ?beaker)
    ;      (and (solution ?solution) (beaker ?beaker))
    ; )

    ; (:stream update_time
    ;     :inputs
    ;     ( ?delta_time)
    ;     :domain(and
    ;         (time ?delta_time)
    ;         ;     ; (time ?delta_time)
    ;     )
    ;     :outputs(?time)
    ;     :certified(and
    ;         (time ?time)
    ;         (time_updated ?time)
    ;     )
    ; )
)