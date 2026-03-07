#include <time.h>
#include "single_snake.h"

int demo() {
    CSnake env = {
        .num_snakes = SINGLE_SNAKE_NUM_SNAKES,
        .width = SINGLE_SNAKE_MAP_WIDTH,
        .height = SINGLE_SNAKE_MAP_HEIGHT,
        .max_snake_length = SINGLE_SNAKE_MAP_WIDTH * SINGLE_SNAKE_MAP_HEIGHT,
        .food = SINGLE_SNAKE_NUM_FOOD,
        .vision = SINGLE_SNAKE_VISION,
        .leave_corpse_on_death = false,
        .reward_food = 1.0f,
        .reward_corpse = 0.0f,
        .reward_death = -1.0f,
        .reward_step = -0.01f,
        .use_potential_shaping = true,
        .potential_shaping_coef = 0.05f,
        .max_episode_steps = SINGLE_SNAKE_MAX_EPISODE_STEPS,
        .cell_size = 48,
    };
    allocate_csnake(&env);
    c_reset(&env);
    env.client = make_client(env.cell_size, env.width, env.height);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 0;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;
        c_step(&env);
        c_render(&env);
    }
    close_client(env.client);
    free_csnake(&env);
    return 0;
}

void test_performance(float test_time) {
    CSnake env = {
        .num_snakes = SINGLE_SNAKE_NUM_SNAKES,
        .width = SINGLE_SNAKE_MAP_WIDTH,
        .height = SINGLE_SNAKE_MAP_HEIGHT,
        .max_snake_length = SINGLE_SNAKE_MAP_WIDTH * SINGLE_SNAKE_MAP_HEIGHT,
        .food = SINGLE_SNAKE_NUM_FOOD,
        .vision = SINGLE_SNAKE_VISION,
        .leave_corpse_on_death = false,
        .reward_food = 1.0f,
        .reward_corpse = 0.0f,
        .reward_death = -1.0f,
        .reward_step = -0.01f,
        .use_potential_shaping = true,
        .potential_shaping_coef = 0.05f,
        .max_episode_steps = SINGLE_SNAKE_MAX_EPISODE_STEPS,
    };
    allocate_csnake(&env);
    c_reset(&env);

    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        for (int j = 0; j < env.num_snakes; j++) {
            env.actions[j] = rand()%4;
        }
        c_step(&env);
        i++;
    }
    int end = time(NULL);
    free_csnake(&env);
    printf("SPS: %f\n", (float)env.num_snakes*i / (end - start));
}

int main() {
    demo();
    // test_performance(30);
    return 0;
}
