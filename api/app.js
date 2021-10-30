const express = require("express");
const cors = require("cors");
const execa = require("execa");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json({ limit: "5mb" }));

app.post("/compute", async (req, res) => {
  const { algo, data, method, arch, h_dim, n_layers, lr, n_epoques, horizon } =
    req.body;
  const input = JSON.stringify({
    algo,
    data,
    method,
    arch,
    h_dim,
    n_layers,
    lr,
    n_epoques,
    horizon,
  });
  await execa.command("python main.py", {
    input,
    forceKillAfterTimeout: 600,
  });
  res.json({ done: true });
});

app.get("/figure.png", (req, res) => {
  res.sendFile(path.join(__dirname, "public/figure.png"));
});

app.post("/upload", async (req, res) => {
  const { value } = req.body;
  const { stdout } = await execa.command("python main.py", { input: value });
  res.send(stdout);
});
app.post("/indicators", async (req, res) => {
  const { value } = req.body;
  res.send(value);
});
app.listen(8000);
