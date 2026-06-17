import type { ProductCard } from "../api/chat";

type ProductCardListProps = {
  productCards: ProductCard[];
};

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
                <span className="price">¥{card.price}</span>
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
                {Object.entries(card.attributes).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
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
