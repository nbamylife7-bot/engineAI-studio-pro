# Running the Demo

After following the installation [instructions](../getting_started/installation.md), the demo can be launched with the commands below. The demo runs in the web browser at [http://localhost:7860](http://localhost:7860).

</details>

<details>
<summary>If you run the demo on a server, you can use port forwarding to access it.</summary>

To access the demo's web interface when running on a remote server, set up SSH port forwarding so your web browser can reach `http://localhost:7860` as if it was local.

**Option 1: Add LocalForward to your SSH config**

Edit (or create) your SSH config file (typically `~/.ssh/config`):

```
Host your-server-name
    HostName your.server.address
    User username
    LocalForward 7860 localhost:7860
```
Then connect with:
```
ssh your-server-name
```

**Option 2: Use the SSH command-line directly**

From your local machine, run:
```
ssh -N -L 7860:localhost:7860 username@your.server.address
```
This will forward your local port 7860 to the server's port 7860.
After connecting, open [`http://localhost:7860`](http://localhost:7860) in your web browser.

Replace `username` and `your.server.address` with your actual user and server info.

</details>
</br>

If you will be restarting the demo frequently, we recommend first starting the text encoder service in the background, as detailed in the [quick start guide](../getting_started/quick_start.md#run-text-encoder-service). If the text encoder service is not running, the demo will automatically load the text encoder model.

The demo will also automatically download the Kimodo model checkpoint on launch and whenever needed when the model preference is changed in the UI.

## Launch from Command Line
If you installed Kimodo as a package or from source, the demo can be started with:
```bash
kimodo_demo
```

## Launch with Docker
If you installed with Docker, you can start the demo with:
```bash
docker compose up demo
```

<details>
<summary>Additional Tips for Docker</summary>

You may find the following commands useful if running Kimodo within the Docker containers. In the example commands below, you can also replace `demo` by `text-encoder`:

**Check logs:**

```bash
docker compose logs demo
```

**Stop service:**

```bash
docker compose stop demo
```

**Restart service:**

```bash
docker compose restart demo
```

**Stop and remove everything:**

```bash
docker compose down
```
