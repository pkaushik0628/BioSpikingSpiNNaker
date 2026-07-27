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
extern int16_lut *tau_x_lookup;
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
static inline post_trace_t initialize_post_trace(void){
    return (post_trace_t) {
        .post1 = 0,
        .post2 = 0,
        .last_spike_time = 0
        };
}

static inline pre_trace_t initialize_pre_trace(void){
    return (pre_trace_t){
        .pre1 = 0
    };
}

/**
Updates the post-synaptic trace: performs exponential decay and increments the trace in case of spike
 */
static inline post_trace_t timing_decay_post(uint32_t current_time, uint32_t last_time, post_trace_t last_trace){
    uint32_t dt = current_time - last_time;

    int32_t decayed_post1 = STDP_FIXED_MUL_16X16(last_trace.post1, maths_lut_exponential_decay(dt, tau_minus_lookup)); //Decay fast trace
    int32_t decayed_post2 = STDP_FIXED_MUL_16X16(last_trace.post2, maths_lut_exponential_decay(dt, tau_y_lookup)); //Decay slow trace
    return (post_trace_t) {.post1 = decayed_post1, .post2 = decayed_post2,.last_spike_time = last_trace.last_spike_time};
}

static inline pre_trace_t timing_decay_pre(uint32_t current_time, uint32_t last_time, pre_trace_t last_trace){
    uint32_t dt = current_time - last_time;

    int32_t decayed_pre1 = STDP_FIXED_MUL_16X16(last_trace.pre1, maths_lut_exponential_decay(dt, tau_plus_lookup)); //Decay fast trace
    return (pre_trace_t) {.pre1 = decayed_pre1};
}

static inline post_trace_t timing_add_post_spike(uint32_t time, uint32_t last_time, post_trace_t last_trace, uint16_t invtau_fast, uint16_t invtau_slow){
    post_trace_t temp = timing_decay_post(time, last_time, last_trace);
    temp.post1 = temp.post1 + invtau_fast;
    temp.post2 = temp.post2 + invtau_slow;
    temp.last_spike_time = time;
    return temp;
}

static inline pre_trace_t timing_add_pre_spike(uint32_t time, uint32_t last_time, pre_trace_t last_trace, uint16_t invtau_fast){
    pre_trace_t temp = timing_decay_pre(time, last_time, last_trace);
    temp.pre1 = temp.pre1 + invtau_fast;
    return temp;
}

static inline update_state_t timing_apply_pre_spike(uint32_t time, UNUSED uint32_t last_pre_time, UNUSED pre_trace_t last_pre_trace, uint32_t last_post_time, post_trace_t last_post_trace, update_state_t previous_state, double dep_amp, double learning_rate){
    uint32_t dt = time - last_post_time;
    int32_t decayed_post = STDP_FIXED_MUL_16X16(last_post_trace.post1, maths_lut_exponential_decay(dt, tau_minus_lookup));
    double gain = (double)(dep_amp * learning_rate);
    double val = gain*(double)decayed_post;
    return weight_one_term_apply_depression(previous_state, (int32_t) val);  
}

static inline update_state_t timing_apply_post_spike(uint32_t time, uint32_t last_pre_time, pre_trace_t last_pre_trace, UNUSED uint32_t last_post_time, post_trace_t last_post_trace, update_state_t previous_state, double pot_amp, double learning_rate){
    uint32_t dt = time - last_pre_time;
    int32_t decayed_pre_fast = STDP_FIXED_MUL_16X16(last_pre_trace.pre1, maths_lut_exponential_decay(dt, tau_plus_lookup));
    int32_t decayed_post_slow = STDP_FIXED_MUL_16X16(last_post_trace.post2, maths_lut_exponential_decay(dt, tau_y_lookup));
    int32_t triplet_term =STDP_FIXED_MUL_16X16(decayed_pre_fast,decayed_post_slow);
    double gain = (double)(pot_amp * learning_rate);
    double val = gain*(double)triplet_term;
    return weight_one_term_apply_potentiation(previous_state, (int32_t)val); 
}

#endif

