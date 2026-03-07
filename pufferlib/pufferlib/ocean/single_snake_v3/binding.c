#include "single_snake_v3.h"

typedef struct _object PyObject;
static PyObject* vec_render_ansi(PyObject* self, PyObject* args);
#define MY_METHODS {"vec_render_ansi", vec_render_ansi, METH_VARARGS, "Render vector env as ANSI text"}

#define Env CSnakeV3
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->reward_food = unpack(kwargs, "reward_food");
    env->reward_step = unpack(kwargs, "reward_step");
    env->reward_death = unpack(kwargs, "reward_death");
    env->max_episode_steps = unpack(kwargs, "max_episode_steps");

    init_csnake(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

static char tile_to_ascii(Env* env, int pos) {
    int head_pos = env->snake[env->head_ptr];
    if (pos == head_pos) {
        return '@';
    }

    switch (env->grid[pos]) {
        case EMPTY: return ' ';
        case FOOD: return '*';
        case SNAKE_TILE: return 'o';
        default: return '?';
    }
}

static PyObject* vec_render_ansi(PyObject* self, PyObject* args) {
    int num_args = PyTuple_Size(args);
    if (num_args < 1 || num_args > 3) {
        PyErr_SetString(PyExc_TypeError, "vec_render_ansi requires 1-3 arguments: vec_handle, env_id=0, stride=1");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    int env_id = 0;
    if (num_args >= 2) {
        PyObject* env_id_arg = PyTuple_GetItem(args, 1);
        if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
            return NULL;
        }
        env_id = PyLong_AsLong(env_id_arg);
    }
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    int stride = 1;
    if (num_args >= 3) {
        PyObject* stride_arg = PyTuple_GetItem(args, 2);
        if (!PyObject_TypeCheck(stride_arg, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "stride must be an integer");
            return NULL;
        }
        stride = PyLong_AsLong(stride_arg);
    }
    if (stride < 1) {
        stride = 1;
    }

    Env* env = vec->envs[env_id];
    int out_w = (env->width + stride - 1) / stride;
    int out_h = (env->height + stride - 1) / stride;
    size_t cap = (size_t)out_h * (size_t)(out_w + 1) + 1;

    char* buffer = (char*)malloc(cap);
    if (!buffer) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate ANSI render buffer");
        return NULL;
    }

    size_t pos = 0;
    for (int y = 0; y < env->height; y += stride) {
        for (int x = 0; x < env->width; x += stride) {
            int cell_pos = y * env->width + x;
            buffer[pos++] = tile_to_ascii(env, cell_pos);
        }
        buffer[pos++] = '\n';
    }

    PyObject* out = PyUnicode_FromStringAndSize(buffer, pos);
    free(buffer);
    return out;
}
