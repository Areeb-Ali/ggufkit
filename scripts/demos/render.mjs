// Render the demos recorded by scripts/make_demos.py into videos, GIFs and posters.
//
//   node scripts/demos/render.mjs DEMOS_JSON [NAME ...] [--formats=vertical,readme] [--keep-frames]
//
// Each demo is rendered twice from scenes.html:
//   vertical  1080x1920, 30 fps -> build/demos/NAME.mp4 and build/demos/NAME-cover.png
//   readme    800x1000,  15 fps -> assets/demos/NAME.gif and assets/demos/NAME.png
// Needs Playwright (with a Chromium it can launch) and ffmpeg on PATH.

import { mkdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

// Through require, so a global Playwright install is found via NODE_PATH.
const { chromium } = createRequire(import.meta.url)("playwright");

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const SCENES = pathToFileURL(join(HERE, "scenes.html")).href;
const FORMATS = {
  vertical: { width: 1080, height: 1920, scale: 1, fps: 30 },
  readme: { width: 1080, height: 1350, scale: 800 / 1080, fps: 15 },
};

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--") && !a.includes("=")));
const formatArg = args.find((a) => a.startsWith("--formats="));
const positional = args.filter((a) => !a.startsWith("--"));
const [dataPath, ...only] = positional;
if (!dataPath) {
  console.error("usage: node render.mjs DEMOS_JSON [NAME ...] [--formats=vertical,readme]");
  process.exit(2);
}
const data = JSON.parse(readFileSync(dataPath, "utf8"));
const names = only.length ? only : Object.keys(data);
const formats = formatArg ? formatArg.split("=")[1].split(",") : Object.keys(FORMATS);

function ffmpeg(...argv) {
  const r = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...argv], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${argv.join(" ")}`);
}

const mb = (path) => (statSync(path).size / 1e6).toFixed(2);

const browser = await chromium.launch();
try {
  for (const format of formats) {
    const f = FORMATS[format];
    const context = await browser.newContext({
      viewport: { width: f.width, height: f.height },
      deviceScaleFactor: f.scale,
    });
    for (const name of names) {
      const page = await context.newPage();
      await page.goto(SCENES);
      const meta = await page.evaluate(([n, d, fmt]) => window.setup(n, d, fmt), [name, data[name], format]);
      await page.evaluate(() => window.ready());

      const frames = join(ROOT, "build", "demos", "frames", `${name}-${format}`);
      rmSync(frames, { recursive: true, force: true });
      mkdirSync(frames, { recursive: true });
      const count = Math.round(meta.duration * f.fps);
      for (let i = 0; i < count; i++) {
        await page.evaluate((t) => window.renderAt(t), i / f.fps);
        await page.screenshot({ path: join(frames, `${String(i).padStart(5, "0")}.png`) });
      }
      await page.evaluate((t) => window.renderAt(t), meta.poster);

      if (format === "vertical") {
        const out = join(ROOT, "build", "demos");
        await page.screenshot({ path: join(out, `${name}-cover.png`) });
        const mp4 = join(out, `${name}.mp4`);
        ffmpeg(
          "-framerate", String(f.fps), "-i", join(frames, "%05d.png"),
          "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
          "-map", "0:v", "-map", "1:a", "-shortest",
          "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
          "-profile:v", "high", "-r", String(f.fps),
          "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", mp4,
        );
        console.log(`${name}: ${mp4} (${mb(mp4)} MB, ${count} frames)`);
      } else {
        const out = join(ROOT, "assets", "demos");
        mkdirSync(out, { recursive: true });
        await page.screenshot({ path: join(out, `${name}.png`) });
        const gif = join(out, `${name}.gif`);
        ffmpeg(
          "-framerate", String(f.fps), "-i", join(frames, "%05d.png"),
          "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
          "-loop", "0", gif,
        );
        console.log(`${name}: ${gif} (${mb(gif)} MB, ${count} frames)`);
      }
      if (!flags.has("--keep-frames")) rmSync(frames, { recursive: true, force: true });
      await page.close();
    }
    await context.close();
  }
} finally {
  await browser.close();
}
