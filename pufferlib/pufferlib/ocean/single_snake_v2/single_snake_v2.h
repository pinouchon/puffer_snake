#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define EMPTY 0
#define FOOD 1
#define SNAKE_TILE 2

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct CSnakeV2 CSnakeV2;
struct CSnakeV2 {
    char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    Log log;
    Log snake_log;

    unsigned char* grid;
    int* snake; // Circular buffer of cell ids in [0, width*height)
    int snake_length;
    int head_ptr;

    int width;
    int height;
    int area;
    int num_food;
    int max_episode_steps;
    unsigned int rng_state;

    float reward_food;
    float reward_step;
    float reward_death;

    unsigned char use_potential_shaping;
    unsigned char shape_on_eat;
    float potential_shaping_coef;
};

static inline int clamp_positive(int v, int min_v) {
    return v < min_v ? min_v : v;
}

static inline unsigned int rng_next(CSnakeV2* env) {
    unsigned int x = env->rng_state;
    if (x == 0) {
        x = 0x9E3779B9u;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng_state = x;
    return x;
}

static inline int rng_bounded(CSnakeV2* env, int bound) {
    if (bound <= 1) {
        return 0;
    }
    return (int)(rng_next(env) % (unsigned int)bound);
}

static inline int pos_to_r(CSnakeV2* env, int pos) {
    return pos / env->width;
}

static inline int pos_to_c(CSnakeV2* env, int pos) {
    return pos % env->width;
}

static inline int rc_to_pos(CSnakeV2* env, int r, int c) {
    return r * env->width + c;
}

static inline void clear_log(Log* log) {
    *log = (Log){0};
}

static inline void add_log(CSnakeV2* env) {
    env->log.perf += env->snake_log.perf;
    env->log.score += env->snake_log.score;
    env->log.episode_return += env->snake_log.episode_return;
    env->log.episode_length += env->snake_log.episode_length;
    env->log.n += 1;
}

static inline int ring_prev(CSnakeV2* env, int ptr) {
    return (ptr - 1 + env->area) % env->area;
}

static inline int ring_tail(CSnakeV2* env) {
    return (env->head_ptr - env->snake_length + 1 + env->area) % env->area;
}

static inline void configure(CSnakeV2* env) {
    env->width = clamp_positive(env->width, 2);
    env->height = clamp_positive(env->height, 2);
    env->num_food = 1;
    env->area = env->width * env->height;

    if (env->max_episode_steps < 0)
        env->max_episode_steps = 0;

}

void init_csnake(CSnakeV2* env) {
    configure(env);

    env->grid = (unsigned char*)calloc(env->area, sizeof(unsigned char));
    env->snake = (int*)calloc(env->area, sizeof(int));

    for (int i = 0; i < env->area; i++) {
        env->snake[i] = -1;
    }

    env->snake_length = 0;
    env->head_ptr = 0;
    env->rng_state = 0x9E3779B9u ^ (unsigned int)rand();
    clear_log(&env->log);
    clear_log(&env->snake_log);
}

void c_close(CSnakeV2* env) {
    free(env->grid);
    free(env->snake);
}

static inline void clear_snake(CSnakeV2* env) {
    for (int i = 0; i < env->snake_length; i++) {
        int ptr = (env->head_ptr - i + env->area) % env->area;
        int pos = env->snake[ptr];
        if (pos >= 0 && pos < env->area)
            env->grid[pos] = EMPTY;
        env->snake[ptr] = -1;
    }

    env->snake_length = 0;
    env->head_ptr = 0;
}

static inline bool spawn_food(CSnakeV2* env) {
    int empty_count = 0;
    for (int pos = 0; pos < env->area; pos++) {
        if (env->grid[pos] == EMPTY) {
            empty_count++;
        }
    }

    if (empty_count == 0) {
        return false;
    }

    int target = rng_bounded(env, empty_count);
    for (int pos = 0; pos < env->area; pos++) {
        if (env->grid[pos] != EMPTY) {
            continue;
        }

        if (target == 0) {
            env->grid[pos] = FOOD;
            return true;
        }
        target--;
    }

    return false;
}

static inline void ensure_single_food(CSnakeV2* env) {
    int first_food = -1;
    for (int pos = 0; pos < env->area; pos++) {
        if (env->grid[pos] != FOOD) {
            continue;
        }

        if (first_food == -1) {
            first_food = pos;
        } else {
            env->grid[pos] = EMPTY;
        }
    }

    if (first_food == -1) {
        (void)spawn_food(env);
    }
}

static inline void spawn_snake(CSnakeV2* env) {
    clear_snake(env);

    int empty_count = 0;
    for (int i = 0; i < env->area; i++) {
        if (env->grid[i] == EMPTY) {
            empty_count++;
        }
    }
    if (empty_count == 0) {
        return;
    }

    int pos = -1;
    int target = rng_bounded(env, empty_count);
    for (int i = 0; i < env->area; i++) {
        if (env->grid[i] != EMPTY) {
            continue;
        }
        if (target == 0) {
            pos = i;
            break;
        }
        target--;
    }
    if (pos < 0) {
        return;
    }

    env->head_ptr = 0;
    env->snake_length = 1;
    env->snake[env->head_ptr] = pos;
    env->grid[pos] = SNAKE_TILE;
    clear_log(&env->snake_log);
}

static inline void compute_observations(CSnakeV2* env) {
    for (int pos = 0; pos < env->area; pos++) {
        unsigned char tile = env->grid[pos];
        if (tile == FOOD)
            env->observations[pos] = 1;
        else if (tile == SNAKE_TILE)
            env->observations[pos] = 3;
        else
            env->observations[pos] = 0;
    }

    int head_pos = env->snake[env->head_ptr];
    if (head_pos >= 0 && head_pos < env->area)
        env->observations[head_pos] = 2;
}

void c_reset(CSnakeV2* env) {
    configure(env);
    env->rng_state = 0x9E3779B9u ^ (unsigned int)rand();

    clear_log(&env->log);
    clear_log(&env->snake_log);
    env->terminals[0] = 0;

    for (int i = 0; i < env->area; i++) {
        env->grid[i] = EMPTY;
        env->snake[i] = -1;
    }

    spawn_snake(env);
    ensure_single_food(env);

    compute_observations(env);
}

static inline void end_episode(CSnakeV2* env, float terminal_reward) {
    env->terminals[0] = 1;
    env->rewards[0] = terminal_reward;
    env->snake_log.episode_return += terminal_reward;
    env->snake_log.score = env->snake_length;
    env->snake_log.perf = env->snake_log.score / env->snake_log.episode_length;
    add_log(env);
    spawn_snake(env);
    ensure_single_food(env);
}

void c_step(CSnakeV2* env) {
    env->terminals[0] = 0;
    env->snake_log.episode_length += 1;

    if (env->max_episode_steps > 0 && env->snake_log.episode_length >= env->max_episode_steps) {
        end_episode(env, env->reward_step);
        compute_observations(env);
        return;
    }

    if (env->snake_length >= env->area) {
        end_episode(env, env->reward_step);
        compute_observations(env);
        return;
    }

    int atn = env->actions[0];
    int dr = 0;
    int dc = 0;
    switch (atn) {
        case 0: dr = -1; break; // up
        case 1: dr = 1; break;  // down
        case 2: dc = -1; break; // left
        case 3: dc = 1; break;  // right
        default: break;
    }

    int old_head_pos = env->snake[env->head_ptr];
    int old_head_r = pos_to_r(env, old_head_pos);
    int old_head_c = pos_to_c(env, old_head_pos);

    int next_r = old_head_r + dr;
    int next_c = old_head_c + dc;

    // Prevent directly reversing into the neck
    if (env->snake_length > 1) {
        int neck_pos = env->snake[ring_prev(env, env->head_ptr)];
        if (next_r == pos_to_r(env, neck_pos) && next_c == pos_to_c(env, neck_pos)) {
            next_r = old_head_r - dr;
            next_c = old_head_c - dc;
        }
    }

    if (next_r < 0 || next_r >= env->height || next_c < 0 || next_c >= env->width) {
        end_episode(env, env->reward_death);
        compute_observations(env);
        return;
    }

    int next_pos = rc_to_pos(env, next_r, next_c);
    unsigned char tile = env->grid[next_pos];
    bool grow = (tile == FOOD);

    if (tile == SNAKE_TILE) {
        int tail_pos = env->snake[ring_tail(env)];
        bool moving_into_tail = !grow && next_pos == tail_pos;
        if (!moving_into_tail) {
            end_episode(env, env->reward_death);
            compute_observations(env);
            return;
        }
    }

    int new_head_ptr = (env->head_ptr + 1) % env->area;
    env->snake[new_head_ptr] = next_pos;
    env->head_ptr = new_head_ptr;

    float reward = env->reward_step;
    if (grow) {
        reward += env->reward_food;
        if (env->snake_length < env->area) {
            env->snake_length++;
        }
    } else {
        int tail_ptr = (env->head_ptr - env->snake_length + env->area) % env->area;
        int tail_pos = env->snake[tail_ptr];
        if (tail_pos >= 0 && tail_pos < env->area) {
            env->grid[tail_pos] = EMPTY;
            env->snake[tail_ptr] = -1;
        }
    }
    env->grid[next_pos] = SNAKE_TILE;

    if (grow && env->snake_length >= env->area) {
        end_episode(env, reward);
        compute_observations(env);
        return;
    }

    ensure_single_food(env);

    env->rewards[0] = reward;
    env->snake_log.episode_return += reward;

    compute_observations(env);
}

void c_render(CSnakeV2* env) {
    (void)env;
}
