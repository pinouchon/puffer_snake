#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "raylib.h"

#define EMPTY 0
#define FOOD 1
#define CORPSE 2
#define WALL 3

#define SINGLE_SNAKE_NUM_SNAKES 1
#define SINGLE_SNAKE_NUM_FOOD 2
#define SINGLE_SNAKE_MAP_WIDTH 10
#define SINGLE_SNAKE_MAP_HEIGHT 10
#define SINGLE_SNAKE_VISION 5
#define SINGLE_SNAKE_MAX_EPISODE_STEPS 400

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct CSnake CSnake;
struct CSnake {
    char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    Log* snake_logs;
    char* grid;
    int* snake;
    int* snake_lengths;
    int* snake_ptr;
    int* snake_lifetimes;
    int* snake_colors;
    int num_snakes;
    int width;
    int height;
    int max_snake_length;
    int food;
    int vision;
    int window;
    int obs_size;
    int border;
    int grid_width;
    int grid_height;
    unsigned char leave_corpse_on_death;
    float reward_food;
    float reward_corpse;
    float reward_death;
    float reward_step;
    unsigned char use_potential_shaping;
    float potential_shaping_coef;
    int max_episode_steps;
    int tick;
    int cell_size;
    Client* client;
};

static inline int grid_index(CSnake* env, int row, int col) {
    return row*env->grid_width + col;
}

void add_log(CSnake* env, int snake_id) {
    env->log.perf += env->snake_logs[snake_id].perf;
    env->log.score += env->snake_logs[snake_id].score;
    env->log.episode_return += env->snake_logs[snake_id].episode_return;
    env->log.episode_length += env->snake_logs[snake_id].episode_length;
    env->log.n += 1;
}

int nearest_food_distance(CSnake* env, int r, int c) {
    int best = env->width + env->height + 1;
    for (int rr = env->border; rr < env->border + env->height; rr++) {
        for (int cc = env->border; cc < env->border + env->width; cc++) {
            if (env->grid[grid_index(env, rr, cc)] != FOOD)
                continue;

            int d = abs(rr - r) + abs(cc - c);
            if (d < best)
                best = d;
        }
    }

    if (best == env->width + env->height + 1)
        return -1;
    return best;
}

void configure_single_snake(CSnake* env) {
    env->num_snakes = SINGLE_SNAKE_NUM_SNAKES;
    env->food = SINGLE_SNAKE_NUM_FOOD;
    env->width = SINGLE_SNAKE_MAP_WIDTH;
    env->height = SINGLE_SNAKE_MAP_HEIGHT;
    env->vision = SINGLE_SNAKE_VISION;
    env->border = env->vision;
    env->grid_width = env->width + 2*env->border;
    env->grid_height = env->height + 2*env->border;

    int max_area = env->width * env->height;
    if (env->max_snake_length <= 0 || env->max_snake_length > max_area)
        env->max_snake_length = max_area;
}

void init_csnake(CSnake* env) {
    configure_single_snake(env);
    env->grid = (char*)calloc(env->grid_width*env->grid_height, sizeof(char));
    env->snake = (int*)calloc(env->num_snakes*2*env->max_snake_length, sizeof(int));
    env->snake_lengths = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_ptr = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_lifetimes = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_colors = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_logs = (Log*)calloc(env->num_snakes, sizeof(Log));
    env->tick = 0;
    env->client = NULL;
    env->snake_colors[0] = 7;
    for (int i = 1; i<env->num_snakes; i++)
        env->snake_colors[i] = i%4 + 4; // Randomize snake colors
}

void c_close(CSnake* env) {
    free(env->grid);
    free(env->snake);
    free(env->snake_lengths);
    free(env->snake_ptr);
    free(env->snake_lifetimes);
    free(env->snake_colors);
    free(env->snake_logs);
}
void allocate_csnake(CSnake* env) {
    configure_single_snake(env);
    int obs_size = env->width * env->height;
    env->observations = (char*)calloc(env->num_snakes*obs_size, sizeof(char));
    env->actions = (int*)calloc(env->num_snakes, sizeof(int));
    env->rewards = (float*)calloc(env->num_snakes, sizeof(float));    
    init_csnake(env);
}

void free_csnake(CSnake* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
}

void compute_observations(CSnake* env) {
    for (int i = 0; i < env->num_snakes; i++) {
        int snake_offset = i*2*env->max_snake_length;
        int head_ptr = env->snake_ptr[i];
        int head_offset = snake_offset + 2*head_ptr;
        int head_r = env->snake[head_offset];
        int head_c = env->snake[head_offset + 1];
        int r_offset = env->border;
        int c_offset = env->border;
        for (int r = 0; r < env->height; r++) {
            for (int c = 0; c < env->width; c++) {
                int rr = r_offset + r;
                int cc = c_offset + c;
                int tile = env->grid[grid_index(env, rr, cc)];
                char obs_tile = EMPTY; // 0: empty

                if (tile == FOOD) {
                    obs_tile = FOOD; // 1: food
                } else if (tile == env->snake_colors[i]) {
                    // 2: own head, 3: own body
                    obs_tile = (rr == head_r && cc == head_c) ? 2 : 3;
                }

                env->observations[i*env->obs_size + r*env->window + c] = obs_tile;
            }
        }
    }
}

void delete_snake(CSnake* env, int snake_id) {
    while (env->snake_lengths[snake_id] > 0) {
        int head_ptr = env->snake_ptr[snake_id];
        int head_offset = 2*env->max_snake_length*snake_id + 2*head_ptr;
        int head_r = env->snake[head_offset];
        int head_c = env->snake[head_offset + 1];
        if (env->leave_corpse_on_death && env->snake_lengths[snake_id] % 2 == 0)
            env->grid[grid_index(env, head_r, head_c)] = CORPSE;
        else
            env->grid[grid_index(env, head_r, head_c)] = EMPTY;

        env->snake[head_offset] = -1;
        env->snake[head_offset + 1] = -1;
        env->snake_lengths[snake_id]--;
        if (head_ptr == 0)
            env->snake_ptr[snake_id] = env->max_snake_length - 1;
        else
            env->snake_ptr[snake_id]--;
    }
}

void spawn_snake(CSnake* env, int snake_id) {
    int head_r, head_c, tile, grid_idx;
    delete_snake(env, snake_id);
    do {
        head_r = env->border + rand() % env->height;
        head_c = env->border + rand() % env->width;
        grid_idx = grid_index(env, head_r, head_c);
        tile = env->grid[grid_idx];
    } while (tile != EMPTY && tile != CORPSE);
    int snake_offset = 2*env->max_snake_length*snake_id;
    env->snake[snake_offset] = head_r;
    env->snake[snake_offset + 1] = head_c;
    env->snake_lengths[snake_id] = 1;
    env->snake_ptr[snake_id] = 0;
    env->snake_lifetimes[snake_id] = 0;
    env->grid[grid_idx] = env->snake_colors[snake_id];
    env->snake_logs[snake_id] = (Log){0};
}

void spawn_food(CSnake* env) {
    int idx = -1;
    int tile = WALL;
    int attempts = env->width * env->height * 4;
    while (attempts-- > 0) {
        int r = env->border + rand() % env->height;
        int c = env->border + rand() % env->width;
        idx = grid_index(env, r, c);
        tile = env->grid[idx];
        if (tile == EMPTY || tile == CORPSE)
            break;
    }
    if (idx < 0 || (tile != EMPTY && tile != CORPSE))
        return;
    env->grid[idx] = FOOD;
}

void c_reset(CSnake* env) {
    configure_single_snake(env);
    env->window = env->width;
    env->obs_size = env->width * env->height;
    env->tick = 0;
    env->log = (Log){0};
    
    for (int i = 0; i < env->num_snakes; i++) {
        env->snake_logs[i] = (Log){0};
        env->terminals[i] = 0;
    }

    int grid_area = env->grid_width * env->grid_height;
    for (int i = 0; i < grid_area; i++)
        env->grid[i] = WALL;

    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int rr = env->border + r;
            int cc = env->border + c;
            env->grid[grid_index(env, rr, cc)] = EMPTY;
        }
    }

    for (int i = 0; i < env->num_snakes; i++)
        spawn_snake(env, i);
    for (int i = 0; i < env->food; i++)
        spawn_food(env);

    compute_observations(env);
}

void step_snake(CSnake* env, int i) {
    env->snake_logs[i].episode_length += 1;

    if (env->max_episode_steps > 0 &&
            env->snake_logs[i].episode_length >= env->max_episode_steps) {
        env->terminals[i] = 1;
        env->rewards[i] = env->reward_step;
        env->snake_logs[i].episode_return += env->rewards[i];
        env->snake_logs[i].score = env->snake_lengths[i];
        env->snake_logs[i].perf = env->snake_logs[i].score / env->snake_logs[i].episode_length;
        add_log(env, i);
        spawn_snake(env, i);
        return;
    }

    int atn = env->actions[i];
    int dr = 0;
    int dc = 0;
    switch (atn) {
        case 0: dr = -1; break; // up
        case 1: dr = 1; break;  // down
        case 2: dc = -1; break; // left
        case 3: dc = 1; break;  // right
    }

    int head_ptr = env->snake_ptr[i];
    int snake_offset = 2*env->max_snake_length*i;
    int head_offset = snake_offset + 2*head_ptr;
    int old_head_r = env->snake[head_offset];
    int old_head_c = env->snake[head_offset + 1];
    int next_r = dr + old_head_r;
    int next_c = dc + old_head_c;

    // Disallow moving into own neck
    int prev_head_offset = head_offset - 2;
    if (prev_head_offset < 0)
        prev_head_offset += 2*env->max_snake_length;
    int prev_r = env->snake[prev_head_offset];
    int prev_c = env->snake[prev_head_offset + 1];
    if (prev_r == next_r && prev_c == next_c) {
        next_r = env->snake[head_offset] - dr;
        next_c = env->snake[head_offset + 1] - dc;
    }

    int tile = env->grid[grid_index(env, next_r, next_c)];
    if (tile >= WALL) {
        env->terminals[i] = 1;
        env->rewards[i] = env->reward_death;
        env->snake_logs[i].episode_return += env->reward_death;
        env->snake_logs[i].score = env->snake_lengths[i];
        env->snake_logs[i].perf = env->snake_logs[i].score / env->snake_logs[i].episode_length;
        add_log(env, i);
        spawn_snake(env, i);
        return;
    }

    head_ptr++; // Circular buffer
    if (head_ptr >= env->max_snake_length)
        head_ptr = 0;
    head_offset = snake_offset + 2*head_ptr;
    env->snake[head_offset] = next_r;
    env->snake[head_offset + 1] = next_c;
    env->snake_ptr[i] = head_ptr;
    env->snake_lifetimes[i]++;

    bool grow;
    float reward = env->reward_step;
    if (tile == FOOD) {
        reward += env->reward_food;
        spawn_food(env);
        grow = true;
    } else if (tile == CORPSE) {
        reward += env->reward_corpse;
        grow = true;
    } else {
        grow = false;
    }
    int snake_length = env->snake_lengths[i];
    if (grow && snake_length < env->max_snake_length - 1) {
        env->snake_lengths[i]++;
    } else {
        int tail_ptr = head_ptr - snake_length;
        if (tail_ptr < 0) // Circular buffer
            tail_ptr = env->max_snake_length + tail_ptr;
        int tail_r = env->snake[snake_offset + 2*tail_ptr];
        int tail_c = env->snake[snake_offset + 2*tail_ptr + 1];
        int tail_offset = 2*env->max_snake_length*i + 2*tail_ptr;
        env->snake[tail_offset] = -1;
        env->snake[tail_offset + 1] = -1;
        env->grid[grid_index(env, tail_r, tail_c)] = EMPTY;
    }
    env->grid[grid_index(env, next_r, next_c)] = env->snake_colors[i];

    if (env->use_potential_shaping) {
        int d_prev = nearest_food_distance(env, old_head_r, old_head_c);
        int d_next = nearest_food_distance(env, next_r, next_c);
        if (d_prev >= 0 && d_next >= 0)
            reward += env->potential_shaping_coef * (d_prev - d_next);
    }

    env->rewards[i] = reward;
    env->snake_logs[i].episode_return += reward;
}

void c_step(CSnake* env){
    env->tick++;
    for (int i = 0; i < env->num_snakes; i++) {
        env->terminals[i] = 0;
        step_snake(env, i);
    }

    compute_observations(env);
}

// Raylib client
Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255},
    (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255},
    (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255},
    (Color){0, 255, 255, 255},
    (Color){255, 255, 0, 255},
};

typedef struct Client Client;
struct Client {
    int cell_size;
    int width;
    int height;
};

Client* make_client(int cell_size, int width, int height) {
    Client* client= (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->width = width;
    client->height = height;
    InitWindow(width*cell_size, height*cell_size, "PufferLib Snake");
    SetTargetFPS(10);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(CSnake* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    if (env->client == NULL) {
        env->client = make_client(env->cell_size, env->width, env->height);
    }
    
    Client* client = env->client;
    
    BeginDrawing();
    ClearBackground(COLORS[0]);
    int sz = client->cell_size;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++){
            int tile = env->grid[grid_index(env, y + env->border, x + env->border)];
            if (tile != EMPTY)
                DrawRectangle(x*sz, y*sz, sz, sz, COLORS[tile]);
        }
    }
    EndDrawing();
}
