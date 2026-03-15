"""
Extremely Simple Order Matching Engine for Limit Order Book
"""

class Order:
    def __init__(self, id, side, price, quantity):
        self.id = id
        self.side = side
        self.price = price
        self.quantity = quantity
        
class LimitOrderBook:
    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.trades = []
        
    def add_order(self, order):
        if order.side == "buy":
            # match bids against asks
            while order.quantity > 0 and self.sell_orders:
                best_ask = self.sell_orders[0]
                if order.price >= best_ask.price:
                    trade_quantity = min(order.quantity, best_ask.quantity)
                    print(f"TRADE: Sell {trade_quantity} at {best_ask.price}")
                    self.trades.append({
                        "buy_order_id": order.id,
                        "sell_order_id": best_ask.id,
                        "price": best_ask.price,
                        "quantity": trade_quantity})
                    order.quantity -= trade_quantity
                    best_ask.quantity -= trade_quantity
                    if best_ask.quantity == 0:
                        self.sell_orders.pop(0)
                else:
                    break
            if order.quantity > 0:
                self.buy_orders.append(order)
                self.buy_orders.sort(key=lambda x: x.price, reverse=True)
                print(f"RESTING: Buy {order.quantity} at {order.price}")
        else: # sell order
            while order.quantity > 0 and self.buy_orders:
                best_bid = self.buy_orders[0]
                if order.price <= best_bid.price:
                    trade_quantity = min(order.quantity, best_bid.quantity)
                    print(f"TRADE: Buy {trade_quantity} at {best_bid.price}")
                    self.trades.append({
                        "buy_order_id": best_bid.id,
                        "sell_order_id": order.id,
                        "price": best_bid.price,
                        "quantity": trade_quantity})
                    order.quantity -= trade_quantity
                    best_bid.quantity -= trade_quantity
                    if best_bid.quantity == 0:
                        self.buy_orders.pop(0)
                else:
                    break
            if order.quantity > 0:
                self.sell_orders.append(order)
                self.sell_orders.sort(key=lambda x: x.price)
                print(f"RESTING: Sell {order.quantity} at {order.price}")
        
book = LimitOrderBook()

print("1. Sell 100 @ $50.15")
book.add_order(Order(1, "sell", 50.15, 100))

print("2. Sell 200 @ $50.20")
book.add_order(Order(2, "sell", 50.20, 200))

print("3. Buy 50 @ $50.10 (no match)")
book.add_order(Order(3, "buy", 50.10, 50))

print("4. Buy 150 @ $50.20 (crosses the spread)")
book.add_order(Order(4, "buy", 50.20, 150))