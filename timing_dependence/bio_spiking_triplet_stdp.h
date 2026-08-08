#ifndef _BIO_SPIKING_TRIPLET_STDP_H_
#define _BIO_SPIKING_TRIPLET_STDP_H_

#include <neuron/plasticity/stdp/synapse_structure/synapse_structure_weight_impl.h>
#include "timing.h"
#include <neuron/plasticity/stdp/weight_dependence/weight_additive_one_term_impl.h>
#include <debug.h>
#include <neuron/plasticity/stdp/maths.h>
#include <neuron/plasticity/stdp/stdp_typedefs.h>

// External Variables
extern int16_lut *tau_plus_lookup;
extern int16_lut *tau_minus_lookup;
extern int16_lut *tau_y_lookup;

/**
Creates a struct to store a post-synaptic trace
post1 = fast trace
post2 = slow trace
last_spike_time = time of the last post synaptic spike
 */
typedef struct post_trace_t {
    int16_t post1; //fast trace
    int16_t post2; //slow trace
    uint32_t last_spike_time;
} post_trace_t;

/**
Creates a struct to store pre-synaptic traces
pre1 = first presynaptic trace
pre2 = second presynaptic trace
 */
typedef struct pre_trace_t {
    int16_t pre1;
} pre_trace_t;

/**
Initializes post_trace_t
Sets the fast and slow traces to 0
 */
static inline post_trace_t timing_get_initial_post_trace(void){
    return (post_trace_t) {
        .post1 = 0,
        .post2 = 0
        };
}

/**
Initializes post_trace_t
Sets the trace to 0
 */
static inline pre_trace_t timing_get_initial_pre_trace(void){
    return (pre_trace_t){
        .pre1 = 0
    };
}

/**
Updates the post-synaptic traces (fast and slow): performs exponential decay to post synaptic trace
@param current time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return decayed/updated symmetric trace
@math U_fast (post) = U_fast*exp(-dt/tau_minus) and U_slow (post) = U_slow*exp(-dt/tau_y)
 */
static inline post_trace_t timing_decay_post(uint32_t current_time, uint32_t last_time, post_trace_t last_trace){
    uint32_t dt = current_time - last_time;

    int32_t decayed_post1 = STDP_FIXED_MUL_16X16(last_trace.post1, maths_lut_exponential_decay(dt, tau_minus_lookup)); //Decay fast trace
    int32_t decayed_post2 = STDP_FIXED_MUL_16X16(last_trace.post2, maths_lut_exponential_decay(dt, tau_y_lookup)); //Decay slow trace
    return (post_trace_t) {.post1 = decayed_post1, .post2 = decayed_post2,.last_spike_time = last_trace.last_spike_time};
}

/**
Updates the pre-synaptic trace: performs exponential decay to pre synaptic trace
@param current time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return decayed/updated pre trace
@math U_fast (pre) = U_fast*exp(-dt/tau_plus)
 */
static inline pre_trace_t timing_decay_pre(uint32_t current_time, uint32_t last_time, pre_trace_t last_trace){
    uint32_t dt = current_time - last_time;

    int32_t decayed_pre1 = STDP_FIXED_MUL_16X16(last_trace.pre1, maths_lut_exponential_decay(dt, tau_plus_lookup)); //Decay fast trace
    return (pre_trace_t) {.pre1 = decayed_pre1};
}

/**
Adds the impact of post spike to STDP trace: performs exponential decay to post synaptic trace and increments the post trace on spike
@param time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return updated post traces (fast and slow)
@math U_slow (post) = U_slow*exp(-dt/tau) + invtau_slow and U_fast (post) = U_fast*exp(-dt/tau) + invtau_fast
 */
static inline post_trace_t timing_add_post_spike(uint32_t time, uint32_t last_time, post_trace_t last_trace){
    post_trace_t temp = timing_decay_post(time, last_time, last_trace);
    temp.post1 = temp.post1 + 5*STDP_FIXED_POINT_ONE;
    temp.post2 = temp.post2 + 2*STDP_FIXED_POINT_ONE;
    temp.last_spike_time = time;
    return temp;
}

/**
Adds the impact of pre spike to STDP trace: performs exponential decay to pre synaptic trace and increments the STDP trace
@param time current simulation time
@param last_time last time of a recorded spike
@param last_trace trace to be updated over dt = time - last_time
@return updated pre traces (fast only)
@math U_fast (pre) = U_fast*exp(-dt/tau) + invtau_fast
 */
static inline pre_trace_t timing_add_pre_spike(uint32_t time, uint32_t last_time, pre_trace_t last_trace){
    pre_trace_t temp = timing_decay_pre(time, last_time, last_trace);
    temp.pre1 = temp.pre1 + 5*STDP_FIXED_POINT_ONE;
    return temp;
}

/**
Called on pre-spike: Updates STDP state by calculating weight updates (depression)
@param time current time t
@param pre_trace current pre trace
@param last_pre_time the previous time of pre-spike
@param last_pre_trace the previous pre-syn trace
@param last_post_time the previous time of post-spike
@param last_post_trace the previous post-syn trace
@param previous_state the state to be updated
@math dW = -A_minus*U_fast_post(t-dt)*exp(-dt/tau), where A_dep = depression amplitude, eta = learning rate
 */
static inline update_state_t timing_apply_pre_spike(uint32_t time, UNUSED pre_trace_t pre_trace, UNUSED uint32_t last_pre_time, UNUSED pre_trace_t last_pre_trace, uint32_t last_post_time, post_trace_t last_post_trace, update_state_t previous_state){
    uint32_t dt = time - last_post_time;
    int32_t decayed_post = STDP_FIXED_MUL_16X16(last_post_trace.post1, maths_lut_exponential_decay(dt, tau_minus_lookup));
    return weight_one_term_apply_depression(previous_state, decayed_post);  
}

/**
Called on post spike: Updates STDP state by calculating weight updates (potentiation)
@param time current time t
@param post_trace current post trace
@param last_pre_time timing of last pre-syn spike
@param last_pre_trace last/un-updated pre-syn trace
@param last_post_time the previous time of post-spike
@param last_post_trace the previous post-syn trace
@param previous_state the state to be updated
@math dW = +A_plus*U_pre_fast(t-dt)*exp(-dt/tau_fast)*U_post_slow(t-dt)*exp(-dt/tau_slow)
 */
static inline update_state_t timing_apply_post_spike(uint32_t time, UNUSED post_trace_t post_trace, uint32_t last_pre_time, pre_trace_t last_pre_trace, UNUSED uint32_t last_post_time, post_trace_t last_post_trace, update_state_t previous_state){
    uint32_t dt = time - last_pre_time;
    int32_t decayed_pre_fast = STDP_FIXED_MUL_16X16(last_pre_trace.pre1, maths_lut_exponential_decay(dt, tau_plus_lookup));
    int32_t decayed_post_slow = STDP_FIXED_MUL_16X16(last_post_trace.post2, maths_lut_exponential_decay(dt, tau_y_lookup));
    int32_t triplet_term =STDP_FIXED_MUL_16X16(decayed_pre_fast,decayed_post_slow);
    return weight_one_term_apply_potentiation(previous_state, triplet_term); 
}

#endif

