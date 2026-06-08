export type HealthResponse = {
  status: string;
  app: string;
  version: string;
};

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch("http://localhost:8000/health");

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<HealthResponse>;
}
