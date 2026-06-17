import type { ProductCard } from "../api/chat";

type ProductCardListProps = {
  productCards: ProductCard[];
};

const ATTRIBUTE_LABELS: Record<string, string> = {
  os: "系统",
  network_type: "网络",
  chipset: "处理器",
  ram_gb: "运行内存",
  storage_gb: "存储容量",
  battery_mah: "电池容量",
  fast_charge_w: "快充功率",
  screen_size_in: "屏幕尺寸",
  refresh_rate_hz: "刷新率",
  rear_camera_max_mp: "后置主摄",
  front_camera_mp: "前置镜头",
  nfc: "NFC",
  ir_blaster: "红外",
  gender: "适用人群",
  style: "风格",
  upper_material: "鞋面材质",
  sole_material: "鞋底材质",
  season: "季节",
  anti_slip: "防滑",
  breathable: "透气",
  sizes: "尺码",
  skin_type: "肤质",
  texture: "质地",
  ingredients: "核心成分",
  contains_fragrance: "香精",
  contains_alcohol: "酒精",
  routine_step: "护肤步骤",
};

const ATTRIBUTE_ORDER = Object.keys(ATTRIBUTE_LABELS);
const INTERNAL_ATTRIBUTE_KEYS = new Set([
  "currency",
  "source_product_id",
  "source_platform",
  "data_quality",
  "rag_product_text",
]);

export function ProductCardList({ productCards }: ProductCardListProps) {
  return (
    <section className="panel">
      <h2>Product Cards</h2>
      {productCards.length > 0 ? (
        <div className="product-list">
          {productCards.map((card) => (
            <article className="product-card" key={card.product_id}>
              <div className="product-header">
                <h3>{card.title}</h3>
                <span className="price">
                  {formatPrice(card.price, card.attributes.currency)}
                </span>
              </div>
              <p className="muted">{card.brand ?? "Unknown brand"}</p>
              <p>{card.recommend_reason}</p>

              <div className="tag-list">
                {card.tags.map((tag) => (
                  <span className="tag" key={tag}>
                    {tag}
                  </span>
                ))}
              </div>

              <dl className="attribute-list">
                {displayAttributes(card).map(({ key, label, value }) => (
                  <div key={key}>
                    <dt>{label}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>

              <div className="link-row">
                {card.source_url ? (
                  <a href={card.source_url} target="_blank" rel="noreferrer">
                    source_url
                  </a>
                ) : null}
                {card.compare_url ? (
                  <a href={card.compare_url} target="_blank" rel="noreferrer">
                    compare_url
                  </a>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">No product cards</p>
      )}
    </section>
  );
}

function displayAttributes(card: ProductCard) {
  const entries = Object.entries(card.attributes ?? {})
    .filter(([key, value]) => isDisplayableAttribute(key, value))
    .sort(([left], [right]) => {
      const leftIndex = ATTRIBUTE_ORDER.indexOf(left);
      const rightIndex = ATTRIBUTE_ORDER.indexOf(right);
      return normalizeSortIndex(leftIndex) - normalizeSortIndex(rightIndex);
    })
    .slice(0, 12)
    .map(([key, value]) => ({
      key,
      label: ATTRIBUTE_LABELS[key] ?? key,
      value: formatAttributeValue(key, value),
    }));

  return entries.length > 0
    ? entries
    : [{ key: "empty", label: "参数", value: "暂无关键参数" }];
}

function isDisplayableAttribute(key: string, value: unknown): boolean {
  if (INTERNAL_ATTRIBUTE_KEYS.has(key)) {
    return false;
  }

  if (!(key in ATTRIBUTE_LABELS)) {
    return false;
  }

  if (value === null || value === undefined || value === "") {
    return false;
  }

  const stringValue = String(value).trim();
  if (!stringValue || stringValue === "null" || stringValue === "undefined") {
    return false;
  }

  return true;
}

function formatAttributeValue(key: string, value: unknown): string {
  const normalized = String(value).trim();

  if (normalized === "True" || normalized === "true") {
    return "支持";
  }

  if (normalized === "False" || normalized === "false") {
    return "不支持";
  }

  if (["ram_gb", "storage_gb"].includes(key)) {
    return appendUnit(normalized, "GB");
  }

  if (key === "battery_mah") {
    return appendUnit(normalized, "mAh");
  }

  if (key === "fast_charge_w") {
    return appendUnit(normalized, "W");
  }

  if (key === "screen_size_in") {
    return appendUnit(normalized, "英寸");
  }

  if (key === "refresh_rate_hz") {
    return appendUnit(normalized, "Hz");
  }

  if (["rear_camera_max_mp", "front_camera_mp"].includes(key)) {
    return appendUnit(normalized, "MP");
  }

  return normalized;
}

function appendUnit(value: string, unit: string): string {
  return value.toLowerCase().includes(unit.toLowerCase())
    ? value
    : `${value}${unit}`;
}

function formatPrice(price: number, currency: unknown): string {
  const normalizedCurrency = String(currency ?? "CNY").toUpperCase();

  if (normalizedCurrency === "CNY" || normalizedCurrency === "RMB") {
    return `¥${price}`;
  }

  if (normalizedCurrency === "USD") {
    return `$${price}`;
  }

  if (normalizedCurrency === "INR") {
    return `₹${price}`;
  }

  return `${price} ${normalizedCurrency}`;
}

function normalizeSortIndex(index: number): number {
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}
