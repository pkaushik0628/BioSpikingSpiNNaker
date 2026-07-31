#ifndef _BIO_SPIKING_SYMMETRIC_STDP_H_
#define _BIO_SPIKING_SYMMETRIC_STDP_H_

#include <neuron/plasticity/stdp/synapse_structure/synapse_structure_weight_impl.h>
#include "timing.h"
#include <neuron/plasticity/stdp/weight_dependence/weight_two_term.h>
#include <debug.h>
#include <neuron/plasticity/stdp/maths.h>
#include <neuron/plasticity/stdp/stdp_typedefs.h>

// External Variables
extern int16_lut *tau_plus_lookup;

/**
Post trace
 */
typedef struct post_trace_t{
    int32_t trace;
    uint32_t last_spike_time;
} post_trace_t;

/**
Pre-trace
 */
typedef struct pre_trace_t{
    int32_t trace;
    uint32_t last_spike_time;
} pre_trace_t;

/**
Initializes a post trace. Sets the post trace to 0
 */
static inline post_trace_t initialize_post_trace(void){
    return (post_trace_t){
        .trace = 0
    };
}

/**
Initializes a pre trace. Sets the pre trace to 0
 */
static inline pre_trace_t initialize_pre_trace(void){
    return (pre_trace_t){
        .trace = 0
    };
}

/**
Updates the post-synaptic trace: performs exponential decay to post synaptic trace
@param current time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return decayed/updated symmetric trace
 */
static inline post_trace_t timing_decay_post(uint32_t current_time, uint32_t last_time, post_trace_t last_trace){
    uint32_t dt = current_time - last_time;
    int32_t decayed_trace = STDP_FIXED_MUL_16X16(last_trace.trace, maths_lut_exponential_decay(dt, tau_symm_lookup));
    return (post_trace_t) {.trace = decayed_trace, .last_spike_time = last_trace.last_spike_time};
}

/**
Updates the pre-synaptic trace: performs exponential decay to pre synaptic trace
@param current time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return decayed/updated symmetric trace
 */
static inline pre_trace_t timing_decay_pre(uint32_t current_time, uint32_t last_time, pre_trace_t last_trace){
    uint32_t dt = current_time - last_time;
    int32_t decayed_trace = STDP_FIXED_MUL_16X16(last_trace.trace, maths_lut_exponential_decay(dt, tau_symm_lookup));
    return (pre_trace_t) {.trace = decayed_trace, .last_spike_time = last_trace.last_spike_time};
}

/**
Adds the impact of post spike to STDP trace: performs exponential decay to post synaptic trace and increments the STDP trace
@param time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return decayed/updated post trace
@math U_post = U_post*exp(-dt/tau) + invtau, where U_post is post_synaptic trace
 */
static inline post_trace_t timing_add_post_spike(uint32_t time, uint32_t last_time, post_trace_t last_trace){
    post_trace_t temp = timing_decay_post(time, last_time, last_trace);
    temp.trace = temp.trace + 5*STDP_FIXED_POINT_ONE;
    temp.last_spike_time = time;
    return temp;
}

/**
Adds the impact of a pre-spike to STDP trace: performs exponential decay to pre synaptic trace and increments the STDP trace
@param time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return decayed/updated pre trace
@math U_pre = U_pre*exp(-dt/tau) + invtau, where U_pre is pre_synaptic trace
 */
static inline pre_trace_t timing_add_pre_spike(uint32_t time, uint32_t last_time, pre_trace_t last_trace){
    pre_trace_t temp = timing_decay_pre(time, last_time, last_trace);
    temp.trace = temp.trace + 5*STDP_FIXED_POINT_ONE;
    temp.last_spike_time = time;
    return temp;
}

/**
Called on pre-spike: Updates STDP state by calculating weight updates (technically depression)
Note: We use weight_one_term_apply_potentiation() because this is potentiation only STDP
@param time current time t
@param pre_trace current pre trace
@param last_pre_time the previous time of pre-spike
@param last_pre_trace the previous pre-syn trace
@param last_post_time the previous time of post-spike
@param last_post_trace the previous post-syn trace
@param previous_state the state to be updated
@math dW = A_minus*U_post(t-dt)*exp(-dt/tau_minus), where A_minus = A_plus, tau_plus = tau_minus
 */
static inline update_state_t timing_apply_pre_spike(uint32_t time, UNUSED pre_trace_t pre_trace, UNUSED uint32_t last_pre_time, UNUSED pre_trace_t last_pre_trace,  uint32_t last_post_time, post_trace_t last_post_trace, update_state_t previous_state){
    uint32_t dt = time - last_post_time;
    int32_t decayed_post = STDP_FIXED_MUL_16X16(last_post_trace.trace, maths_lut_exponential_decay(dt, tau_plus_lookup));
    return weight_one_term_apply_potentiation(previous_state, decayed_post);  
}

/**
Called on post spike: Updates STDP state by calculating weight updates (technically potentiation)
@param time current time t
@param post_trace current post trace
@param last_pre_time timing of last pre-syn spike
@param last_pre_trace last/un-updated pre-syn trace
@param last_post_time the previous time of post-spike
@param last_post_trace the previous post-syn trace
@param previous_state the state to be updated
@math dW = A_plus*U_pre(t-dt)*exp(-dt/tau_plus), where A_plus = A_minus, tau_plus = tau_minus
 */
static inline update_state_t timing_apply_post_spike(uint32_t time, UNUSED post_trace_t post_trace, uint32_t last_pre_time, pre_trace_t last_pre_trace, UNUSED uint32_t last_post_time, UNUSED post_trace_t last_post_trace, update_state_t previous_state){
    uint32_t dt = time - last_pre_time;
    int32_t decayed_pre = STDP_FIXED_MUL_16X16(last_pre_trace.trace, maths_lut_exponential_decay(dt, tau_plus_lookup));
    return weight_one_term_apply_potentiation(previous_state, decayed_pre); 
}

