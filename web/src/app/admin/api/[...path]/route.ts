// The admin page's server-side path to the api's internal admin endpoints (design §22.1). The
// browser never reaches /internal/* on the api (next.config.ts proxies /api/v1/* only); this
// route forwards a fixed set of admin paths, and only when the web server runs with
// RECLOOP_ADMIN=1 (`make serve-web ADMIN=1`). Everything else is a 404: not exposed.

const apiUrl = process.env.RECLOOP_API_URL ?? "http://127.0.0.1:18080";
const ALLOWED = /^(models(\/[A-Za-z0-9._-]+\/(validate|canary|activate))?|canary\/abort|rollback|experiments(\/[A-Za-z0-9._-]+(\/stop)?)?|fault)$/;

async function forward(request: Request, params: Promise<{ path: string[] }>): Promise<Response> {
  if (process.env.RECLOOP_ADMIN !== "1") {
    return Response.json({ error: "the admin api is off (start the web app with RECLOOP_ADMIN=1)" }, { status: 404 });
  }
  const path = (await params).path.join("/");
  if (!ALLOWED.test(path)) {
    return Response.json({ error: "not an admin path" }, { status: 404 });
  }
  const body = request.method === "GET" ? undefined : await request.text();
  const upstream = await fetch(`${apiUrl}/internal/admin/${path}`, {
    method: request.method,
    headers: { "Content-Type": "application/json" },
    body: body || undefined,
    cache: "no-store",
  });
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function GET(request: Request, { params }: { params: Promise<{ path: string[] }> }) {
  return forward(request, params);
}

export async function POST(request: Request, { params }: { params: Promise<{ path: string[] }> }) {
  return forward(request, params);
}

export async function PUT(request: Request, { params }: { params: Promise<{ path: string[] }> }) {
  return forward(request, params);
}
