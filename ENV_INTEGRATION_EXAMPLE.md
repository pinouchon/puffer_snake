# Orderbook Env Integration in PufferLib

This document explains how `orderbook` was added to this repo and how it plugs into PufferLib's Ocean environment system.

## High-level integration path

The integration has four layers:

1. Native env engine (`orderbook.h`) implements env state + step/reset/render/close.
2. Python C extension (`binding.c`) exposes that engine through the shared `env_binding.h` contract.
3. Python env wrapper (`orderbook.py`) gives a Gym-like `PufferEnv` interface.
4. Env registration (`environment.py`) maps `puffer_orderbook` to `Orderbook` so `env_creator(...)` can construct it.

A config file (`config/ocean/orderbook.ini`) sets training/runtime defaults for this env.

## File structure and responsibilities

- `pufferlib/ocean/orderbook/orderbook.h`
- Contains the core exchange simulation and RL env logic.
- Defines `struct Orderbook` (the native env state), `struct Log` (metrics), and core lifecycle functions (`init`, `c_reset`, `c_step`, `c_render`, `c_close`).

- `pufferlib/ocean/orderbook/binding.c`
- Binds native code to Python.
- Defines `#define Env Orderbook` and `#define MY_METHODS ...`, then includes `../env_binding.h`.
- Implements `my_init(...)` (reads kwargs into native env fields), `my_log(...)` (maps native log fields to Python dict), and custom exchange APIs (`place_limit_order`, `cancel_all`, `get_level_one_data`, etc.).

- `pufferlib/ocean/orderbook/orderbook.py`
- Python-side RL env class: `class Orderbook(pufferlib.PufferEnv)`.
- Allocates/uses shared buffers from `PufferEnv`, creates one native env per Python env, then vectorizes handles via `binding.vectorize(*c_envs)`.
- Exposes `reset`, `step`, `render`, `close` in expected PufferLib/Ocean style.

- `pufferlib/ocean/orderbook/exchange_api.py`
- Optional direct exchange control wrapper (`ExchangeEnv`) on top of `binding` for programmatic order placement/querying.
- Not required by the RL loop, but useful for scripting/tests/manual interaction.

- `pufferlib/ocean/environment.py`
- Registration point: adds `'orderbook': 'Orderbook'` to `MAKE_FUNCTIONS` so `env_creator('puffer_orderbook', ...)` resolves this env.

- `pufferlib/config/ocean/orderbook.ini`
- Training/runtime defaults (env args + PPO/train settings) for `puffer_orderbook`.

## Build/packaging contract

The env is auto-built because `setup.py` discovers all Ocean bindings with:

- `glob.glob('pufferlib/ocean/**/binding.c', recursive=True)`

So adding `pufferlib/ocean/orderbook/binding.c` is enough to include this env's extension in normal builds.

## API and contracts used

## 1) Ocean env discovery contract

`pufferlib.ocean.environment.env_creator(...)` expects env names like `puffer_<name>`.

For this env:

- Input name: `puffer_orderbook`
- Internal key: `orderbook`
- Registry lookup in `MAKE_FUNCTIONS`
- Module import path: `pufferlib.ocean.orderbook.orderbook`
- Class/function name from registry: `Orderbook`

So this env is integrated with the same discovery path as other Ocean envs.

## 2) Python `PufferEnv` contract

`orderbook.py` follows the common Ocean wrapper pattern:

- Sets `self.num_agents`, observation space, action space.
- Calls `super().__init__(buf=buf)` to get shared numpy buffers.
- Passes slices of those buffers into `binding.env_init(...)`.
- Returns Gym-style tuples from `reset` and `step`.

Important shape/dtype assumptions in this wrapper:

- `actions` are cast to `float32` (`self.actions = self.actions.astype(np.float32)`).
- Observation size is fixed (`self.num_obs = 5 + 4*5 + 4 + 6` = `35`).
- Discrete action mode uses `MultiDiscrete([3, 11, 11, 2])`.
- Continuous mode uses `Box(shape=(4,))`.

## 3) Shared C binding contract (`env_binding.h`)

`env_binding.h` is the reusable ABI layer used by Ocean envs. `orderbook/binding.c` satisfies that contract by:

- Defining `Env` as the native struct (`Orderbook`).
- Implementing env-specific hooks:
  - `my_init(Env* env, PyObject* args, PyObject* kwargs)`
  - `my_log(PyObject* dict, Log* log)`
- Providing native lifecycle fns used by generic wrappers:
  - `c_reset(env)`, `c_step(env)`, `c_render(env)`, `c_close(env)`

`env_binding.h` then provides the generic Python API surface:

- Single-env: `env_init`, `env_reset`, `env_step`, `env_render`, `env_close`, `env_get`, `env_put`
- Vectorized: `vectorize`, `vec_reset`, `vec_step`, `vec_log`, `vec_render`, `vec_close`
- Extension hook: `MY_METHODS` (this env uses it to expose orderbook-specific methods)

Buffer and type contracts enforced in `env_binding.h`:

- Observations/actions/rewards/terminals/truncations must be contiguous numpy arrays.
- Rewards/terminals/truncations expected as 1D views for each env slice.
- Action buffer must not be float64 (explicit runtime check).
- `env_init` takes: `(obs, actions, rewards, terminals, truncations, seed, **kwargs)`.

## 4) Orderbook-specific native API contract

Beyond standard env lifecycle, `orderbook/binding.c` exports custom exchange calls through `MY_METHODS`, including:

- `place_limit_order(env_handle, trader_id, is_buy, price, qty)`
- `place_market_order(...)`
- `cancel_latest(...)`, `cancel_n(...)`, `cancel_all(...)`
- `get_level_one_data(env_handle)`
- `get_level_two_data(env_handle)`
- `get_all_market_data(env_handle)`
- `get_trader_data(env_handle, trader_id)`

This is why `exchange_api.py` can present a higher-level trading API while still using the same native env.

## Runtime data flow

1. Training config chooses `env_name = puffer_orderbook`.
2. `env_creator` resolves `orderbook -> Orderbook`.
3. `Orderbook.__init__` allocates spaces/buffers and calls `binding.env_init(...)` per env.
4. `binding.my_init` copies kwargs into native `Orderbook` fields, then calls `init(env)`.
5. Training loop writes actions into shared numpy buffers.
6. `binding.vec_step(...)` calls `c_step(...)` for each native env.
7. Native env writes observations/rewards/terminals directly into shared buffers.
8. `vec_log(...)` aggregates native `Log` structs and `my_log(...)` maps them to Python dict keys.

## Practical checklist for adding another Ocean env (same pattern)

1. Create `pufferlib/ocean/<env>/<env>.h` (or similar native implementation) with `init/c_reset/c_step/c_render/c_close` and `Log` fields.
2. Create `pufferlib/ocean/<env>/binding.c` with:
   - `#define Env <YourStruct>`
   - `my_init`, `my_log`
   - optional `MY_METHODS`
   - `#include "../env_binding.h"`
3. Create Python wrapper `pufferlib/ocean/<env>/<env>.py` inheriting `pufferlib.PufferEnv`.
4. Register in `pufferlib/ocean/environment.py` `MAKE_FUNCTIONS`.
5. Add config in `pufferlib/config/ocean/<env>.ini`.
6. Build/install; `setup.py` auto-discovers `binding.c` and compiles it.

## Notes specific to this implementation

- `orderbook.c` appears to be a standalone native test/demo/perf harness and is separate from the extension build path (build is driven by `binding.c`).
- The env supports both discrete and continuous action formats, plus additional orderbook-specific read/write APIs for non-RL usage.