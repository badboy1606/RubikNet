#include <stdio.h>
#include <string.h>
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define STEPS_90 50  


void step_motor(gpio_num_t step_pin, gpio_num_t dir_pin, int clockwise) {
    gpio_set_level(dir_pin, clockwise ? 1 : 0);
    for (int i = 0; i < STEPS_90; i++) {
        gpio_set_level(step_pin, 1);
        ets_delay_us(800);   // adjust speed
        gpio_set_level(step_pin, 0);
        ets_delay_us(800);
    }
}

void rotate_R(int clockwise) {
   
    step_motor(GPIO_NUM_19, GPIO_NUM_21, clockwise);
}

void rotate_L(int clockwise) {
    
    step_motor(GPIO_NUM_18, GPIO_NUM_22, !clockwise);
}

void rotate_U(int clockwise) {
    step_motor(GPIO_NUM_26, GPIO_NUM_23, clockwise);
}

void rotate_D(int clockwise) {
    step_motor(GPIO_NUM_25, GPIO_NUM_27, !clockwise); 
}

void rotate_F(int clockwise) {
    step_motor(GPIO_NUM_13, GPIO_NUM_32, clockwise);
}

void rotate_B(int clockwise) {
    step_motor(GPIO_NUM_12, GPIO_NUM_33, !clockwise); 
}


void app_main(void) {
   
    gpio_reset_pin(GPIO_NUM_19); gpio_set_direction(GPIO_NUM_19, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_21); gpio_set_direction(GPIO_NUM_21, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_18); gpio_set_direction(GPIO_NUM_18, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_22); gpio_set_direction(GPIO_NUM_22, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_26); gpio_set_direction(GPIO_NUM_26, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_23); gpio_set_direction(GPIO_NUM_23, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_25); gpio_set_direction(GPIO_NUM_25, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_27); gpio_set_direction(GPIO_NUM_27, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_13); gpio_set_direction(GPIO_NUM_13, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_32); gpio_set_direction(GPIO_NUM_32, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_12); gpio_set_direction(GPIO_NUM_12, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_33); gpio_set_direction(GPIO_NUM_33, GPIO_MODE_OUTPUT);

    char move[20];
    while (1) {
        int len = scanf("%19s", move);
        if (len > 0) {
            int clockwise = (strchr(move, '\'') == NULL);

           
            switch (move[0]) {
                case 'R': rotate_R(clockwise); break;
                case 'L': rotate_L(clockwise); break;
                case 'U': rotate_U(clockwise); break;
                case 'D': rotate_D(clockwise); break;
                case 'F': rotate_F(clockwise); break;
                case 'B': rotate_B(clockwise); break;
                default: break;
            }
        }
    }
}