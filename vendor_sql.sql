CREATE DATABASE inventory_project;
SHOW DATABASES;
USE inventory_project;
USE inventory_project;

SELECT * FROM purchases LIMIT 10;

-- {"“Analyzed vendor purchasing patterns, inventory movement, product demand, freight cost efficiency, and profit margin opportunities using SQL.”}

-- Top Vendors by Total Purchase Value:
-- Helps identify which vendors the company spends the most money on.

SELECT 
    VendorName,
    SUM(Dollars) AS total_purchase_value,
    SUM(Quantity) AS total_items_purchased
FROM purchases
GROUP BY VendorName
ORDER BY total_purchase_value DESC
LIMIT 10;

-- Business insight:
-- Shows which vendors contribute the largest share of purchasing cost.

-- Inventory Change (Beginning vs Ending Inventory):
-- Tracks inventory movement during the period.

SELECT 
    b.Store,
    SUM(b.onHand) AS beginning_inventory,
    SUM(e.onHand) AS ending_inventory,
    SUM(e.onHand) - SUM(b.onHand) AS inventory_change
FROM begin_inventory b
JOIN end_inventory e 
ON b.InventoryId = e.InventoryId
GROUP BY b.Store
ORDER BY inventory_change DESC;

-- Business insight:
                  -- Shows which stores gained or lost inventory.
                  
-- Most Purchased Products :
                         -- Finds high-demand products.
                         
SELECT 
    Description,
    SUM(Quantity) AS total_quantity_purchased,
    SUM(Dollars) AS total_purchase_cost
FROM purchases
GROUP BY Description
ORDER BY total_quantity_purchased DESC
LIMIT 10;


-- Business insight:
                -- Helps identify top-selling products.
                
-- Vendor Freight Cost Analysis
                            -- Measures logistics cost efficiency.
                            
SELECT 
    VendorName,
    SUM(Dollars) AS total_invoice_value,
    SUM(Freight) AS total_freight_cost,
    ROUND(SUM(Freight) / SUM(Dollars) * 100, 2) AS freight_percentage
FROM vendor_invoice
GROUP BY VendorName
ORDER BY freight_percentage DESC;


-- Business insight:
                  -- Shows vendors with unusually high shipping costs.
                  
                  
-- Price Difference Between Purchase Price and Selling Price:
                                                        -- Analyzes profit margin potential.
                                                        
                                                        
SELECT 
    p.Description,
    p.PurchasePrice,
    pp.Price AS retail_price,
    (pp.Price - p.PurchasePrice) AS potential_profit
FROM purchases p
JOIN purchase_prices pp
ON p.Description = pp.Description
ORDER BY potential_profit DESC
LIMIT 10;

-- Business insight:
                  -- Highlights products with the highest profit potential.
                  

                  
-- {"Performed vendor dependency analysis, inventory turnover analysis,
--  and product value segmentation using SQL to identify supply chain risk, high-demand inventory items, and key purchasing drivers."}
 -- ABC Inventory Analysis (Product Importance)
SELECT 
    Description,
    SUM(Dollars) AS total_purchase_value
FROM purchases
GROUP BY Description
ORDER BY total_purchase_value DESC;

-- {"What this query does
-- Groups all purchase records by product description-- Calculates the total money spent on each product
-- Sorts products by highest spending
-- Business insight
-- This helps classify products into:
-- A category → high-value products (top revenue drivers)
-- B category → moderate importance
-- C category → low-value products
--  Businesses focus inventory control on A-category products because they represent most of the cost or revenue.
-- Example insight you could write:
                             -- A small percentage of products account for the majority of purchasing expenditure, indicating high dependency on key inventory items."}
-- Vendor Dependency Analysis
SELECT 
    VendorName,
    COUNT(DISTINCT Description) AS number_of_products,
    SUM(Dollars) AS total_purchase_value
FROM purchases
GROUP BY VendorName
ORDER BY total_purchase_value DESC;

-- {"What this query does
-- Counts how many different products each vendor supplies
-- Calculates total purchase value per vendor
-- Business insight :
                  -- Shows whether the company is dependent on a few vendors.
-- Example findings:
                    -- One vendor supplies many critical products
					-- Some vendors have high purchase value but supply few items

-- Vendor dependency creates supply chain risk.
-- Example insight:
               -- The company relies heavily on a small number of vendors for a large share of purchasing value, which may expose the supply chain to disruption risks."}
               
-- Inventory Turnover Indicator

SELECT 
    b.Description,
    SUM(b.onHand) AS beginning_inventory,
    SUM(e.onHand) AS ending_inventory,
    SUM(b.onHand) - SUM(e.onHand) AS inventory_used
FROM begin_inventory b
JOIN end_inventory e
ON b.InventoryId = e.InventoryId
GROUP BY b.Description
ORDER BY inventory_used DESC;

-- {"What this query does
-- Compares beginning inventory vs ending inventory
-- Calculates how much inventory was used or sold
-- Business insight
-- Identifies:
               -- Fast-moving products
			   -- Slow-moving or excess inventory
-- Example insight:
-- Products with large inventory reduction indicate strong demand, while items with minimal change may signal slow-moving stock.
-- This helps with:
                -- inventory optimization
                -- reducing storage cost
                -- preventing overstock"}
                
                
                
-- Sales vs Inventory Risk Analysis

-- "Developed inventory risk analysis using SQL joins across multiple datasets to estimate product demand and identify high-selling and overstocked items"

SELECT 
    b.Description,
    SUM(b.onHand) AS beginning_inventory,
    SUM(e.onHand) AS ending_inventory,
    SUM(p.Quantity) AS total_purchased,
    (SUM(b.onHand) + SUM(p.Quantity) - SUM(e.onHand)) AS estimated_sales
FROM begin_inventory b
JOIN end_inventory e 
    ON b.InventoryId = e.InventoryId
LEFT JOIN purchases p
    ON b.InventoryId = p.InventoryId
GROUP BY b.Description
ORDER BY estimated_sales DESC
LIMIT 10;

-- {"What This Query Does
-- This query combines three tables:
									-- begin_inventory
                                    -- end_inventory
                                    -- purchases
-- It calculates:
               -- Beginning stock
                -- Ending stock
                -- Purchased quantity
-- Estimated sales
-- The formula used:
               -- Estimated Sales = Beginning Inventory + Purchases − Ending Inventory
-- Business Insight
           -- This helps the company understand:
-- 1️⃣ Fast Selling Products
                         -- Products with high estimated sales are in strong demand.
-- Action:
          -- Increase stock
           -- Avoid stockouts
-- 2️⃣ Overstocked Products
                        -- Products where ending inventory is still very high may not be selling well.
-- Action:
           -- Reduce purchasing
            -- Offer discounts
-- 3️⃣ Inventory Planning
                      -- Helps the company decide:
-- what products to reorder
-- what inventory is slow-moving
-- Why This Query Is Impressive
-- This is closer to real retail analytics because it:
                                                        -- combines multiple datasets
													     -- calculates derived metrics
                                                             -- produces actionable business insight"}
                                                             
-- Pareto Analysis (80/20 Rule) for Inventory Value
                                              -- Businesses often find that 20% of products account for about 80% of the inventory value. This is called the Pareto Principle.
 
 SELECT 
    Description,
    SUM(Dollars) AS total_value,
    RANK() OVER (ORDER BY SUM(Dollars) DESC) AS product_rank
FROM purchases
GROUP BY Description
ORDER BY total_value DESC;


-- {"What This Query Does
-- Calculates total purchase value per product
-- Ranks products from highest value to lowest
-- Identifies the most financially important products
-- It uses a window function (RANK), which is commonly used in real analytics work.
-- Business Insight
                 -- This analysis helps identify:
                        -- High-value products
                        -- Products contributing the most to purchasing cost.
-- Action:
          -- Closely monitor stock levels 
          -- Ensure supply availability
		-- Low-value products-
		-- Products with minimal financial impact.

-- Action:
          -- Reduce storage space
          -- Avoid overstocking
           -- Strategic inventory focus
           -- Businesses should prioritize inventory management on high-value products.

-- Example insight:
                  -- A small number of products contribute to the majority of total purchasing value, indicating strong concentration of inventory investment."}


                                              
                                              
										
                                                             
          -- db_path="invoice vendor project/data/inventory.db.db"                                                    
                                                             