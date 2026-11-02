import numpy as np
import mujoco as mj
import imageio

from compare_plots import compute_metrics, plot_comparison


class Recorder:
    def __init__(self, name, model, data, t_push=2.0, dur=0.1, F=15.0,
                 target_fps=60, video=True):
        self.name, self.data, self.video = name, data, video
        self.t_push, self.dur, self.F = t_push, dur, F
        self.rows = []
        self.writer = None
        # whole number of physics steps per frame -> video time == sim time
        self.steps_per_frame = max(1, round(1.0 / (target_fps * model.opt.timestep)))
        self.fps = 1.0 / (self.steps_per_frame * model.opt.timestep)

    def log(self, u, x_ref=0.0):
        """Call inside controller(), after u is computed."""
        d = self.data
        self.rows.append((d.time, d.qpos[0], d.qpos[1], float(np.ravel(u)[0]), x_ref))

    def grab(self, window, glfw, context):
        """Call right after mj.mjr_render(), before swap_buffers."""
        if not self.video:
            return
        w, h = glfw.get_framebuffer_size(window)
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        mj.mjr_readPixels(rgb, None, mj.MjrRect(0, 0, w, h), context)
        frame = np.flipud(rgb)[: h // 2 * 2, : w // 2 * 2]
        if self.writer is None:
            self.writer = imageio.get_writer(f"sim_{self.name}.mp4", fps=self.fps,
                                             codec="libx264", quality=8,
                                             macro_block_size=1)
        self.writer.append_data(frame)

    def finish(self):
        """Call after glfw.terminate()."""
        if self.writer is not None:
            self.writer.close()
            print(f"Saved video: sim_{self.name}.mp4")
        a = np.array(self.rows)
        run = dict(name=self.name, t=a[:, 0], x=a[:, 1], theta=a[:, 2], u=a[:, 3],
                   ref_x=a[:, 4], t_push=self.t_push, dur=self.dur, F=self.F)
        np.savez(f"results_{self.name}.npz", **run)
        for k, v in compute_metrics(run).items():
            print(f"{k}: {v:.4f}")
        for theme in ("light", "dark"):
            plot_comparison({self.name: run}, theme, prefix=f"single_{self.name}")