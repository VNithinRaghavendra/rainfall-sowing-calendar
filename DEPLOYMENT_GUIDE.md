# AgroCast Deployment Guide: Netlify & Render

Since the AgroCast Sowing Dashboard is designed as a client-side web application (where `index.html` dynamically loads the pre-computed `data.json` database), you can host it **completely for free** on both **Netlify** and **Render**!

Here is the exact step-by-step guide to deploy your project on both platforms.

---

## ⚡ Part 1: Deploying on Netlify (Free Frontend Hosting)

Netlify is one of the premier platforms for hosting static web applications. There are two simple ways to deploy your dashboard:

### Method 1A: Netlify Drop (Instant Drag-and-Drop) - *Fastest*
No code pushes, command lines, or repository connections required:
1. Open your computer's file explorer and locate the dashboard directory:  
   `C:\Users\nithi\.gemini\antigravity\scratch\rainfall_sowing_calendar\dashboard`  
   *(This folder contains `index.html` and `data.json`).*
2. Open your web browser and go to [app.netlify.com/drop](https://app.netlify.com/drop).
3. Drag your local `dashboard` folder and drop it directly into the upload area on the Netlify page.
4. **Done!** Netlify will instantly package the assets and give you a public link (e.g., `https://agrocast.netlify.app`).

### Method 1B: Netlify Git Integration (Continuous Deployment)
If you have pushed your project code to GitHub:
1. Sign in to [netlify.com](https://www.netlify.com/) using your GitHub account.
2. On your dashboard, click **Add new site** -> **Import an existing project**.
3. Select **GitHub** and authorize Netlify. Search for and select your `rainfall-sowing-calendar` repository.
4. In the **Site configuration** page:
   - **Branch to deploy:** `main` (or `master`)
   - **Base directory:** Leave blank (or `/`)
   - **Build command:** Leave completely blank
   - **Publish directory:** Enter **`dashboard`** *(this is critical so Netlify serves the dashboard folder)*
5. Click **Deploy site**.
6. **Done!** Every time you push updates to GitHub, Netlify will automatically rebuild and redeploy your live site!

---

## 🔺 Part 2: Deploying on Render (Free Static Site Hosting)

Render is a modern cloud platform that provides free static site hosting with instant SSL certificates and global CDN distribution.

### Step-by-Step Render Deployment:
1. **GitHub Upload:** Ensure your project code (including the `dashboard` folder) is pushed to a public or private repository on **GitHub** (e.g. `https://github.com/yourusername/rainfall-sowing-calendar`).
2. **Sign Up:** Go to [render.com](https://render.com) and sign in using your **GitHub** account.
3. **Create New Static Site:**
   - In the Render Dashboard, click the blue **New +** button in the top right.
   - Select **Static Site** from the dropdown menu.
4. **Connect Repository:**
   - Find your `rainfall-sowing-calendar` repository in the list and click **Connect**.
5. **Configure Build Settings:**
   - **Name:** Enter a unique name for your project (e.g., `agrocast-sowing-calendar`).
   - **Branch:** `main` (or `master`)
   - **Build Command:** Leave completely blank (or clear it so it's empty, as our site is pure HTML/JS and doesn't require a compilation command).
   - **Publish Directory:** Enter **`dashboard`** *(this tells Render to serve the files inside our dashboard folder as the web root)*.
6. **Deploy:**
   - Click the green **Create Static Site** button at the bottom of the page.
7. **Done!** Render will deploy the site in under a minute and provide a live URL at the top left of the dashboard (e.g., `https://agrocast-sowing-calendar.onrender.com`).
